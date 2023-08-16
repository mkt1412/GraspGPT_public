import argparse
import os
import tqdm
import time
import copy
import sys
import torch
import numpy as np
import torch.nn.functional as F
from models.graspgpt_plain import GraspGPT_plain
from transformers import BertTokenizer, BertModel, logging
from data.SGNLoader import pc_normalize
from config import get_cfg_defaults
from geometry_utils import farthest_grasps, regularize_pc_point_count
from visualize import draw_scene, get_gripper_control_points
logging.set_verbosity_error()

DEVICE = "cuda"
CODE_DIR = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(CODE_DIR)

def encode_text(text, tokenizer, model, device, type=None):
    """
    Language data encoding with a Google pre-trained BERT
    """
    if type == 'od':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=300).to(device)
    elif type == 'td':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=200).to(device)
    elif type == 'li':
        encoded_input = tokenizer(text, return_tensors='pt', padding="max_length", max_length=21).to(device)
    else:
         raise ValueError(f'No such language embedding type: {type}')
    
    with torch.no_grad():
        output = model(**encoded_input)
        word_embedding = output[0]
        sentence_embedding = torch.mean(output[0], dim=1)
    
    return word_embedding, sentence_embedding, encoded_input['attention_mask']

def load_model(cfg):
    """
    Load GraspGPT from checkpoint
    """

    model = GraspGPT_plain(cfg)
    model_weights = torch.load(
        cfg.weight_file,
        map_location=DEVICE)['state_dict']
    
    model.load_state_dict(model_weights)
    model = model.to(DEVICE)
    model.eval()

    return model
    
def test(model, pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask):   

    pc = pc.type(torch.cuda.FloatTensor)
    obj_desc = torch.from_numpy(obj_desc).unsqueeze(0).to(DEVICE)
    obj_desc_mask = torch.from_numpy(obj_desc_mask).unsqueeze(0).to(DEVICE)
    task_desc = torch.from_numpy(task_desc).unsqueeze(0).to(DEVICE)
    task_desc_mask = torch.from_numpy(task_desc_mask).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
    logits = logits.squeeze()
    probs = torch.sigmoid(logits)
    preds = torch.round(probs)
    
    return probs, preds

def load_pc_and_grasps(data_dir, obj_name):
    obj_dir = os.path.join(data_dir, obj_name)

    pc_file = os.path.join(obj_dir, 'fused_pc_clean.npy')
    grasps_file = os.path.join(obj_dir, 'fused_grasps_clean.npy')

    if not os.path.exists(pc_file):
        print('Unaable to find clean pc and grasps ')
        pc_file = os.path.join(obj_dir, 'fused_pc.npy')
        grasps_file = os.path.join(obj_dir, 'fused_grasps.npy')
        if not os.path.exists(pc_file):
            raise ValueError(
                'Unable to find un-processed point cloud file {}'.format(pc_file))

    pc = np.load(pc_file)
    grasps = np.load(grasps_file)

    # Ensure that grasp and pc is mean centered
    pc_mean = pc[:, :3].mean(axis=0)
    pc[:, :3] -= pc_mean
    grasps[:, :3, 3] -= pc_mean

    # number of candidate grasps
    grasps = farthest_grasps(
        grasps, num_clusters=32, num_grasps=min(50, grasps.shape[0]))

    grasp_idx = 0

    pc[:, :3] += pc_mean
    grasps[:, :3, 3] += pc_mean

    return pc, grasps

def main(args, cfg):

    task = args.task  # 'pour'
    obj_class = args.obj_class  # 'saucepan'
    obj_name = args.obj_name  # 'pan'
    data_dir = args.data_dir

    obj_desc_dir = os.path.join(data_dir, 'descriptions', obj_class)
    task_desc_dir = os.path.join(data_dir, 'descriptions', task)

    # load GraspGPT
    model = load_model(cfg)

    # load BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = bert_model.to(DEVICE)
    bert_model.eval()
    
    pc, grasps = load_pc_and_grasps(os.path.join(data_dir, 'pcs'), obj_name)
    pc_input = regularize_pc_point_count(
        pc, cfg.num_points, use_farthest_point=False)

    pc_mean = pc_input[:, :3].mean(axis=0)
    pc_input[:, :3] -= pc_mean  # mean substraction
    grasps[:, :3, 3] -= pc_mean  # mean substraction

    preds = []
    probs = []

    all_grasps_start_time = time.time()

    # natural language instruction
    task_ins_txt = input('\nPlease input a natural language instruction (e.g., grasp the knife to cut): ')
    task_ins, _, task_ins_mask = encode_text(task_ins_txt, tokenizer, bert_model, DEVICE, type='li')

    # eval each grasp in a loop
    for i in tqdm.trange(len(grasps)):
        start = time.time()
        grasp = grasps[i]

        pc_color = pc_input[:, 3:]
        pc = pc_input[:, :3]

        grasp_pc = get_gripper_control_points()
        grasp_pc = np.matmul(grasp, grasp_pc.T).T  # transform grasps
        grasp_pc = grasp_pc[:, :3]  # remove latent indicator

        latent = np.concatenate(
            [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])  # create latent indicator
        latent = np.expand_dims(latent, axis=1)
        pc = np.concatenate([pc, grasp_pc], axis=0)  # [4103, 3]

        pc, grasp = pc_normalize(pc, grasp, pc_scaling=cfg.pc_scaling)
        pc = np.concatenate([pc, latent], axis=1)  # add back latent indicator

        # load language embeddings
        pc = torch.tensor([pc])

        # object class description embeddings
        obj_desc_path =  os.path.join(obj_desc_dir, 'descriptions', str(np.random.randint(0, 10)))
        if not os.path.exists(obj_desc_path):
            raise ValueError(f"No such object description path: {obj_desc_path}")
        obj_desc_txt = open(os.path.join(obj_desc_path, 'all.txt')).readlines()[0]
        obj_desc = np.load(os.path.join(obj_desc_path, 'word_embed.npy'))[0]
        obj_desc_mask = np.load(os.path.join(obj_desc_path, 'attn_mask.npy'))[0]
        # task description embeddings 
        task_desc_path = os.path.join(task_desc_dir, 'descriptions', str(np.random.randint(0, 10)))
        if not os.path.exists(task_desc_path):
            raise ValueError(f"No such task description dir: {task_desc_path}")
        task_desc_txt = open(os.path.join(task_desc_path, 'all.txt')).readlines()[0]
        task_desc = np.load(os.path.join(task_desc_path, 'word_embed.npy'))[0]
        task_desc_mask = np.load(os.path.join(task_desc_path, 'attn_mask.npy'))[0]

        prob, pred = test(model, pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)

        preds.append(pred.tolist())
        probs.append(prob.tolist())
    

    # output a language instruction and two descriptions
    print("\n")
    print(f"Natural language instruction:\n{task_ins_txt}\n")
    print(f"Object class description:\n{obj_desc_txt}\n")
    print(f"Task description:\n{task_desc_txt}\n")

    # visualize top-K prediction 
    print('Inference took {}s for {} grasps'.format(time.time() - all_grasps_start_time, len(grasps)))
    preds = np.array(preds)
    probs = np.array(probs)

    K = 5
    topk_inds = probs.argsort()[-K:][::-1]
    preds = preds[topk_inds]
    probs = probs[topk_inds]
    grasps = grasps[topk_inds]

    grasp_colors = np.stack([np.ones(probs.shape[0]) -
                             probs, probs, np.zeros(probs.shape[0])], axis=1)
    
    draw_scene(
        pc_input,
        grasps,
        grasp_colors=list(grasp_colors),
        max_grasps=len(grasps))

    best_grasp = copy.deepcopy(grasps[np.argmax(probs)])
    draw_scene(pc_input, np.expand_dims(best_grasp, axis=0))


if __name__ == '__main__':
    """
    python gcngrasp/demo.py cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml --obj_name pan --obj_class saucepan --task pour
    python gcngrasp/demo.py cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml --obj_name spatula --obj_class spatula --task scoop
    python gcngrasp/demo.py cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml --obj_name mug --obj_class mug --task drink
    """

    parser = argparse.ArgumentParser(description="visualize data and stuff")
    parser.add_argument('--task', help='', default='scoop')
    parser.add_argument('--obj_class', help='', default='spatula')
    parser.add_argument('--data_dir', help='location of sample data', default='')
    parser.add_argument('--obj_name', help='', default='spatula')
    parser.add_argument(
        '--cfg_file',
        help='yaml file in YACS config format to override default configs',
        default='cfg/eval/gcngrasp/gcngrasp_split_mode_t_split_idx_3_.yml',
        type=str)

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file != '':
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)
        else:
            raise ValueError('Please provide a valid config file for the --cfg_file arg')

    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    cfg.batch_size = 16

    if len(cfg.gpus) == 1:
        torch.cuda.set_device(cfg.gpus[0])

    experiment_dir = os.path.join(cfg.log_dir, cfg.weight_file)

    weight_files = os.listdir(os.path.join(experiment_dir, 'weights'))
    assert len(weight_files) == 1
    cfg.weight_file = os.path.join(experiment_dir, 'weights', weight_files[0])

    if args.data_dir == '':
        args.data_dir = os.path.join(cfg.base_dir, 'sample_data')

    cfg.freeze()
    print(cfg)

    main(args, cfg)
