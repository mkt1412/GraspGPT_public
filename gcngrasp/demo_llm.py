import argparse
import os
import tqdm
import time
import random
import sys
import openai
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
from data_specification import TASKS, OPENAI_API_KEY, OBJ_PROMPTS, TASK_PROMPTS
openai.api_key = OPENAI_API_KEY

def gpt(text):
    """
    OpenAI GPT API
    """
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=text,
    temperature=1.0,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    return response['choices'][0]['text'].strip()

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

def gen_gpt_desc(class_label, task_label):
    """
    Generate object class and task descriptions
    """        
    class_keys = [random.choice(['shape', 'geometry']), random.choice(["use", "func"]), 
    random.choice(["sim_shape", "sim_geo"]), random.choice(["sim_use", "sim_func"])]
    task_keys = [random.choice(['func', 'use']), "sim_effect", random.choice(['sem_verb', 'sim_verb'])]

    print("\nGenerating object class description ......\n")
    class_desc = []
    for c_key in class_keys:
        prompt = OBJ_PROMPTS[c_key]
        prompt = prompt.replace('OBJ_CLASS', class_label)
        temp_ans = gpt(prompt)
        print(f"[{c_key}] "+temp_ans)
        class_desc.append(temp_ans)
        time.sleep(20)
    class_desc = ' '.join(item for item in class_desc)
    
    print("\nGenerating task description ......\n")
    task_desc = []
    for t_key in task_keys:
        prompt = TASK_PROMPTS[t_key]
        prompt = prompt.replace('TASK_CLASS', task_label)
        temp_ans = gpt(prompt)
        print(f"[{t_key}] "+temp_ans)
        task_desc.append(temp_ans)
        time.sleep(20)
    task_desc = ' '.join(item for item in task_desc)

    return class_desc, task_desc

def load_model(cfg):
    """
    Load GraspGPT pre-trained weight from checkpoint
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
        print('Unable to find clean pc and grasps ')
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
    # grasps = farthest_grasps(
    #     grasps, num_clusters=32, num_grasps=min(50, grasps.shape[0]))
    grasps = farthest_grasps(
        grasps, num_clusters=32, num_grasps=min(1, grasps.shape[0]))

    grasp_idx = 0

    pc[:, :3] += pc_mean
    grasps[:, :3, 3] += pc_mean

    return pc, grasps

def main(args, cfg):

    task = args.task  # 'scoop'
    obj_class = args.obj_class  # 'spatula'
    obj_name = args.obj_name  # 'spatula'
    data_dir = args.data_dir

    # load GraspGPT
    model = load_model(cfg)

    # load BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model = bert_model.to(DEVICE)
    bert_model.eval()

    # load point cloud and grasps
    pc, grasps = load_pc_and_grasps(data_dir, obj_name)
    pc_input = regularize_pc_point_count(
        pc, cfg.num_points, use_farthest_point=False)

    pc_mean = pc_input[:, :3].mean(axis=0)
    pc_input[:, :3] -= pc_mean  # mean substraction
    grasps[:, :3, 3] -= pc_mean  # mean substraction

    preds = []
    probs = []

    all_grasps_start_time = time.time()

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

        # language descriptions
        obj_desc_txt, task_desc_txt = gen_gpt_desc(obj_class, task)
        obj_desc, _, obj_desc_mask = encode_text(obj_desc_txt, tokenizer, bert_model, DEVICE, type='od')
        task_desc, _, task_desc_mask = encode_text(task_desc_txt, tokenizer, bert_model, DEVICE, type='td')
        # language instruciton
        task_ins_txt = input('\nPlease input a natural language instruction (e.g., grasp the knife to cut): ')
        task_ins, _, task_ins_mask = encode_text(task_ins_txt, tokenizer, bert_model, DEVICE, type='li')

        prob, pred = test(model, pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)

        preds.append(pred.tolist())
        probs.append(prob.tolist())
    
    # output a language instruction and two descriptions
    print("\n")
    print(f"Natural language instruction:\n{task_ins_txt}\n")
    print(f"Object class description:\n{obj_desc_txt}\n")
    print(f"Task description:\n{obj_desc_txt}\n")

    print('Inference took {}s for {} grasps'.format(time.time() - all_grasps_start_time, len(grasps)))
    preds = np.array(preds)
    probs = np.array(probs)

    # colored with task compatibility score (green is higher)
    grasp_colors = np.stack([np.ones(probs.shape[0]) -
                             probs, probs, np.zeros(probs.shape[0])], axis=1)
    
    # pc and grasp visualization
    draw_scene(
        pc_input,
        grasps,
        grasp_colors=list(grasp_colors),
        max_grasps=len(grasps))


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
        args.data_dir = os.path.join(cfg.base_dir, 'sample_data/pcs')

    cfg.freeze()
    print(cfg)

    main(args, cfg)
