import argparse
import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from config import get_cfg_defaults
from models.graspgpt_plain import GraspGPT_plain
from data.GCNLoader import GCNTaskGrasp
from data.data_specification import TASKS
from utils.splits import get_ot_pairs_taskgrasp

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../'))
from visualize import draw_scene, mkdir

DEVICE = "cuda"

def visualize_batch(pc, grasps):
    """ Visualizes all the data in the batch, just for debugging """

    for i in range(pc.shape[0]):
        pcc = pc[i, :, :3]
        grasp = grasps[i, :, :]
        draw_scene(pcc, [grasp, ])


def visualize_batch_wrong(pc, grasps, labels, preds):
    """ Visualizes incorrect predictions """

    for i in range(pc.shape[0]):
        if labels[i] != preds[i]:
            print('labels {}, prediction {}'.format(labels[i], preds[i]))
            pcc = pc[i, :, :3]
            grasp = grasps[i, :, :]
            draw_scene(pcc, [grasp, ])

def main(cfg, save=False, visualize=False, experiment_dir=None):

    _, _, _, name2wn = pickle.load(
        open(os.path.join(cfg.base_dir, cfg.folder_dir, 'misc.pkl'), 'rb'))  # e.g. 009_pan':'saucepan.n.01', 190 instances -> 75 categories
    class_list = pickle.load(
        open(
            os.path.join(
                cfg.base_dir,
                'class_list.pkl'),
            'rb')) if cfg.use_class_list else list(
        name2wn.values())  # e.g. 000:'squeezer.n.01'

    dset = GCNTaskGrasp(
        cfg.num_points,
        transforms=None,
        train=0,
        base_dir=cfg.base_dir,
        folder_dir=cfg.folder_dir,
        normal=cfg.model.use_normal,
        tasks=TASKS,  # 56
        map_obj2class=name2wn,
        class_list=class_list,
        split_mode=cfg.split_mode,
        split_idx=cfg.split_idx,
        split_version=cfg.split_version,
        pc_scaling=cfg.pc_scaling,
        use_task1_grasps=cfg.use_task1_grasps,
        graph_data_path=cfg.graph_data_path,
        include_reverse_relations=cfg.include_reverse_relations,
        subgraph_sampling=cfg.subgraph_sampling,
        sampling_radius=cfg.sampling_radius,
        instance_agnostic_mode=cfg.instance_agnostic_mode
    )
  
    model = GraspGPT_plain(cfg)

    assert model._class_list == class_list
    model_weights = torch.load(
        cfg.weight_file,
        map_location=DEVICE)['state_dict']

    model.load_state_dict(model_weights)
    model = model.to(DEVICE)
    model.eval()

    dloader = torch.utils.data.DataLoader(
    dset,
    batch_size=cfg.batch_size,
    shuffle=False,
    collate_fn=GCNTaskGrasp.collate_fn)

    all_preds = []
    all_probs = []
    all_labels = []
    all_data_vis = {}
    all_data_pc = {}

    # Only considering Stage 2 grasps: valid class-task pair
    task1_results_file = os.path.join(
        cfg.base_dir, cfg.folder_dir, 'task1_results.txt')
    assert os.path.exists(task1_results_file)

    object_task_pairs = get_ot_pairs_taskgrasp(task1_results_file)
    TASK2_ot_pairs = object_task_pairs['True'] + \
    object_task_pairs['Weak True']
    TASK1_ot_pairs = object_task_pairs['False'] + \
    object_task_pairs['Weak False']

    all_preds_2 = []
    all_probs_2 = []
    all_labels_2 = []

    all_preds_2_v2 = defaultdict(dict)
    all_probs_2_v2 = defaultdict(dict)
    all_labels_2_v2 = defaultdict(dict)

    # Only considering Stage 1 grasps
    all_preds_1 = []
    all_probs_1 = []
    all_labels_1 = []

    print('Running evaluation on Test set')
    with torch.no_grad():
        for batch in tqdm(dloader):

            pc, pc_color, tasks, classes, instances, grasps, labels, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask = batch

            pc = pc.type(torch.cuda.FloatTensor)
            obj_desc = obj_desc.to(DEVICE)
            obj_desc_mask = obj_desc_mask.to(DEVICE)
            task_desc = task_desc.to(DEVICE)
            task_desc_mask = task_desc_mask.to(DEVICE)
            task_ins = task_ins.to(DEVICE)
            task_ins_mask = task_ins_mask.to(DEVICE)

            logits = model(pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
            logits = logits.squeeze()  # [32]

            probs = torch.sigmoid(logits)
            preds = torch.round(probs)  # 1 or 0

            try:
                preds = preds.cpu().numpy()
                probs = probs.cpu().numpy()
                labels = labels.cpu().numpy()

                # append predictions of current batch
                all_preds += list(preds)
                all_probs += list(probs)
                all_labels += list(labels)
            except TypeError:
                all_preds.append(preds.tolist())
                all_probs.append(probs.tolist())
                all_labels.append(labels.tolist()[0])

            tasks = tasks.cpu().numpy()  # target tasks
            instances = instances.cpu().numpy()  # target instances
            for i in range(tasks.shape[0]):
                task = tasks[i]
                task = TASKS[task]  # e.g. "till" 
                instance_id = instances[i]  #  e.g. 151
                obj_instance_name = dset._all_object_instances[instance_id]  # e.g. "039_brush"
                ot = "{}-{}".format(obj_instance_name, task)  # e.g. "039_brush-till"

                # all predictions
                try:
                    pred = preds[i]  # e.g. 0.0
                    prob = probs[i]  # e.g. instance_id
                    label = labels[i]  # e.g. 0.0
                except IndexError:
                    # TODO: This is very hacky, fix it
                    pred = preds.tolist()
                    prob = probs.tolist()
                    label = labels.tolist()[0]

                # if valid task-class pair
                if ot in TASK2_ot_pairs:
                    all_preds_2.append(pred)
                    all_probs_2.append(prob)
                    all_labels_2.append(label)

                    try:
                        all_preds_2_v2[obj_instance_name][task].append(pred)
                        all_probs_2_v2[obj_instance_name][task].append(prob)
                        all_labels_2_v2[obj_instance_name][task].append(label)
                    # for each instance, record result for each task
                    except KeyError:
                        all_preds_2_v2[obj_instance_name][task] = [pred, ]
                        all_probs_2_v2[obj_instance_name][task] = [prob, ]
                        all_labels_2_v2[obj_instance_name][task] = [label, ]
                # invalid task-class pair
                elif ot in TASK1_ot_pairs:
                    all_preds_1.append(pred)
                    all_probs_1.append(prob)
                    all_labels_1.append(label)
                elif ot in ROUND1_GOLD_STANDARD_PROTOTYPICAL_USE:
                    all_preds_2.append(pred)
                    all_probs_2.append(prob)
                    all_labels_2.append(label)

                    try:
                        all_preds_2_v2[obj_instance_name][task].append(pred)
                        all_probs_2_v2[obj_instance_name][task].append(prob)
                        all_labels_2_v2[obj_instance_name][task].append(label)

                    except KeyError:
                        all_preds_2_v2[obj_instance_name][task] = [pred, ]
                        all_probs_2_v2[obj_instance_name][task] = [prob, ]
                        all_labels_2_v2[obj_instance_name][task] = [label, ]

                else:
                    raise Exception('Unknown ot {}'.format(ot))

            if visualize or save:

                pc = pc.cpu().numpy()  # [32, 4103, 3]
                grasps = grasps.cpu().numpy()  # [32, 4, 4]
                classes = classes.cpu().numpy()  # [32,]
                pc_color = pc_color.cpu().numpy()  # [32, 4096, 3]

                # Uncomment the following for debugging
                # visualize_batch(pc, grasps)
                # visualize_batch_wrong(pc, grasps, labels, preds)

                for i in range(pc.shape[0]):
                    pc_i = pc[i, :, :]
                    pc_i = pc_i[np.where(pc_i[:, 3] == 0), :3].squeeze(0)  # object pc only, pc_i[:, 3] == 1 -> gripper pc
                    pc_color_i = pc_color[i, :, :3]
                    pc_i = np.concatenate([pc_i, pc_color_i], axis=1)  # [4096, 6], colored point cloud
                    grasp = grasps[i, :, :]  # [4, 4]
                    task = tasks[i]
                    task = TASKS[task]  # "till"
                    instance_id = instances[i]
                    obj_instance_name = dset._all_object_instances[instance_id]  # "039_brush"
                    obj_class = classes[i]
                    obj_class = class_list[obj_class]  # scrub_brush.n.01

                    try:
                        pred = preds[i]
                        prob = probs[i]
                        label = labels[i]
                    except IndexError:
                        pred = preds.tolist()
                        prob = probs.tolist()
                        label = labels.tolist()[0]

                    ot = "{}-{}".format(obj_instance_name, task)  # '039_brush-till'
                    grasp_datapt = (grasp, prob, pred, label)
                    if ot in all_data_vis:
                        all_data_vis[ot].append(grasp_datapt)
                        all_data_pc[ot] = pc_i
                    else:
                        all_data_vis[ot] = [grasp_datapt, ]
                        all_data_pc[ot] = pc_i

    # Stage 1+2 grasps
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    random_probs = np.random.uniform(low=0, high=1, size=(len(all_probs)))
    results = {
        'preds': all_preds,
        'probs': all_probs,
        'labels': all_labels,
        'random': random_probs}

    # Only Stage 2 grasps
    all_preds_2 = np.array(all_preds_2)
    all_probs_2 = np.array(all_probs_2)
    all_labels_2 = np.array(all_labels_2)
    random_probs_2 = np.random.uniform(low=0, high=1, size=(len(all_probs_2)))
    results_2 = {
        'preds': all_preds_2,
        'probs': all_probs_2,
        'labels': all_labels_2,
        'random': random_probs_2}

    # Only Stage 1 grasps
    all_preds_1 = np.array(all_preds_1)
    all_probs_1 = np.array(all_probs_1)
    all_labels_1 = np.array(all_labels_1)
    random_probs_1 = np.random.uniform(low=0, high=1, size=(len(all_probs_1)))
    results_1 = {
        'preds': all_preds_1,
        'probs': all_probs_1,
        'labels': all_labels_1,
        'random': random_probs_1}

    # Only Stage 2 grasps: for each instance, record result for each task
    random_probs_2 = np.random.uniform(low=0, high=1, size=(len(all_probs_2)))
    results_2_v2 = {
        'preds': all_preds_2_v2,
        'probs': all_probs_2_v2,
        'labels': all_labels_2_v2,
        'random': random_probs_2}

    # TODO: figure out what each result represents
    if save:
        mkdir(os.path.join(experiment_dir, 'results'))
        pickle.dump(
            results,
            open(
                os.path.join(
                    experiment_dir,
                    'results',
                    "results.pkl"),
                'wb'))

        mkdir(os.path.join(experiment_dir, 'results1'))
        pickle.dump(
            results_1,
            open(
                os.path.join(
                    experiment_dir,
                    'results1',
                    "results.pkl"),
                'wb'))

        mkdir(os.path.join(experiment_dir, 'results2'))
        pickle.dump(
            results_2,
            open(
                os.path.join(
                    experiment_dir,
                    'results2',
                    "results.pkl"),
                'wb'))

    if save or visualize:
        mkdir(os.path.join(experiment_dir, 'results2_ap'))
        pickle.dump(
            results_2_v2,
            open(
                os.path.join(
                    experiment_dir,
                    'results2_ap',
                    "results.pkl"),
                'wb'))

        # TODO - Write separate script for loading and visualizing predictions
        # mkdir(os.path.join(experiment_dir, 'visualization_data'))
        # pickle.dump(
        #     all_data_vis,
        #     open(
        #         os.path.join(
        #             experiment_dir,
        #             'visualization_data',
        #             "predictions.pkl"),
        #         'wb'))

    if visualize:

        mkdir(os.path.join(experiment_dir, 'visualization'))
        mkdir(os.path.join(experiment_dir, 'visualization', 'task1'))
        mkdir(os.path.join(experiment_dir, 'visualization', 'task2'))

        print('saving ot visualizations')
        for ot in all_data_vis.keys():

            if ot in TASK1_ot_pairs:
                save_dir = os.path.join(
                    experiment_dir, 'visualization', 'task1')
            elif ot in TASK2_ot_pairs:
                save_dir = os.path.join(
                    experiment_dir, 'visualization', 'task2')
            else:
                continue

            pc = all_data_pc[ot]
            grasps_ot = all_data_vis[ot]
            grasps = [elem[0] for elem in grasps_ot]
            probs = np.array([elem[1] for elem in grasps_ot])
            preds = np.array([elem[2] for elem in grasps_ot])
            labels = np.array([elem[3] for elem in grasps_ot])

            grasp_colors = np.stack(
                [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_colors=list(grasp_colors), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_gt.png'.format(ot)))

            grasp_colors = np.stack(
                [np.ones(preds.shape[0]) - preds, preds, np.zeros(preds.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_colors=list(grasp_colors), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_pred.png'.format(ot)))

            grasp_colors = np.stack(
                [np.ones(probs.shape[0]) - probs, probs, np.zeros(probs.shape[0])], axis=1)
            draw_scene(pc, grasps, grasp_colors=list(grasp_colors), max_grasps=len(
                grasps), save_dir=os.path.join(save_dir, '{}_probs.png'.format(ot)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument(
        'cfg_file',
        help='yaml file in YACS config format to override default configs',
        default='',
        type=str)
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)
    parser.add_argument('--batch_size', default=32, type=int)

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file != '':
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)

    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    cfg.batch_size = args.batch_size
    if args.gpus == -1:
        args.gpus = [0, ]
    cfg.gpus = args.gpus

    experiment_dir = os.path.join(cfg.log_dir, cfg.weight_file)

    weight_files = os.listdir(os.path.join(experiment_dir, 'weights'))
    assert len(weight_files) == 1
    cfg.weight_file = os.path.join(experiment_dir, 'weights', weight_files[0])  # substitute with checkpoint 

    cfg.freeze()
    print(cfg)
    main(
        cfg,
        save=args.save,
        visualize=args.visualize,
        experiment_dir=experiment_dir)
