import argparse
import pickle
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import operator
from sklearn.metrics import average_precision_score
from collections import defaultdict

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../'))
from visualize import mkdir

def get_results_dir(log_dir, results_dir):
    dirs = os.listdir(log_dir)
    for dir in dirs:
        if dir.find(results_dir) >= 0:
            return dir
    raise FileNotFoundError('Unable to find appropriate folder for experiment {}'.format(results_dir))

def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])

def get_ap_random(n_seeds, labels):
    aps = []
    N = len(labels)
    for i in range(n_seeds):
        np.random.seed(i)
        random_probs = np.random.uniform(low=0, high=1, size=(N)).tolist()
        ap = average_precision_score(labels, random_probs)
        aps.append(ap)
    return np.mean(aps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument('--base_dir', default='', help='Location of dataset', type=str)
    parser.add_argument('--log_dir', default='', help='Location of pretrained checkpoint models', type=str)
    parser.add_argument('--save_dir', default='', type=str)
    args = parser.parse_args()

    if args.base_dir != '':
        if not os.path.exists(args.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    args.base_dir))
    else:
        args.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    if args.log_dir != '':
        if not os.path.exists(args.log_dir):
            raise FileNotFoundError(
                'Provided checkpoint dir {} not found'.format(
                    args.log_dir))
    else:
        args.log_dir = os.path.join(os.path.dirname(__file__), '../checkpoints')

    folder_dir = 'taskgrasp'
    _, _, _, map_obj2class = pickle.load(
        open(os.path.join(args.base_dir, folder_dir, 'misc.pkl'), 'rb'))

    if args.save_dir == '':
        args.save_dir = os.path.join(os.path.dirname(__file__), '../results')

    
    # TODO: detect checkpoint folders automatically
    # exp_title = 'Held-out Tasks GCNGrasp'
    # data = {
    #     "0":"gcngrasp_split_mode_t_split_idx_0",
    #     "1":"gcngrasp_split_mode_t_split_idx_1",
    #     "2":"gcngrasp_split_mode_t_split_idx_2",
    #     "3":"gcngrasp_split_mode_t_split_idx_3",
    # }
    # exp_name = 'gcngrasp_t'

    # exp_title = 'Held-out Objects GCNGrasp'
    # data = {
    #     "0":"gcngrasp_split_mode_o_split_idx_0",
    #     "1":"gcngrasp_split_mode_o_split_idx_1",
    #     "2":"gcngrasp_split_mode_o_split_idx_2",
    #     "3":"gcngrasp_split_mode_o_split_idx_3",
    # }
    # exp_name = 'gcngrasp_o'

    exp_title = 'Single checkpoint evaluation'
    data = {
        "0": "gcngrasp_split_mode_t_split_idx_3__2023-04-27-10-20"
    }
    exp_name = 'gcngrasp_split_mode_t_split_idx_3_2023-04-27-10-20'

    args.save_dir = os.path.join(args.save_dir, exp_name)
    mkdir(args.save_dir)

    merged_task_ap = {}
    merged_class_ap = {}
    merged_obj_ap = {}
    merged_task_ap_random = {}
    merged_class_ap_random = {}
    merged_obj_ap_random = {}

    merged_task_labels = []
    merged_class_labels = []
    merged_obj_labels = []

    # Plotting information
    window_title = "mAP"
    x_label = "Average Precision"
    to_show = False
    plot_color = 'royalblue'

    # TODO: modify here sho that i can handle arbitrary weight file
    for split_idx, split_dir in data.items():
        pkl_file = get_results_dir(args.log_dir, split_dir)
        print('Loading {} results from {}'.format(split_idx, pkl_file))
        pkl_file = os.path.join(args.log_dir, pkl_file, 'results2_ap', 'results.pkl')  # load results file
        results = pickle.load(open(pkl_file, 'rb'))

        # object instance level
        obj_ap = defaultdict(list)
        obj_ap_random = defaultdict(list) 
        obj_probs = defaultdict(list)
        obj_labels = defaultdict(list)

        # object class level
        class_ap = defaultdict(list)
        class_ap_random = defaultdict(list)
        class_probs = defaultdict(list)
        class_labels = defaultdict(list)

        # task level
        task_ap = defaultdict(list)
        task_ap_random = defaultdict(list)
        task_probs = defaultdict(list)
        task_labels = defaultdict(list)

        preds = results['preds']
        probs = results['probs']
        labels = results['labels']

        for obj in probs.keys():
            if type(obj) != tuple:
                for task in probs[obj].keys():
                    if type(task) == str:
                        assert len(probs[obj][task]) == len(labels[obj][task])
                        # record instance-level statistics
                        obj_probs[obj] += probs[obj][task]
                        obj_labels[obj] += labels[obj][task]

                        # record task-level statistics
                        task_probs[task] += probs[obj][task]
                        task_labels[task] += labels[obj][task]

                        # record class-lecel statistics
                        obj_class = map_obj2class[obj]
                        class_probs[obj_class] += probs[obj][task]
                        class_labels[obj_class] += labels[obj][task]

        # compute instance ap
        for obj in obj_probs.keys():
            obj_prob = obj_probs[obj]
            obj_label = obj_labels[obj]
            merged_obj_labels += obj_labels[obj]
            assert len(obj_prob) == len(obj_label)
            ap_random = get_ap_random(5, obj_label)
            ap = average_precision_score(obj_label, obj_prob)  # compute ap for this instance
            if not np.isnan(ap):
                obj_ap[obj] = ap
                obj_ap_random[obj] = ap_random

        # compute task ap
        for task in task_probs.keys():
            task_prob = task_probs[task]
            task_label = task_labels[task]
            merged_task_labels += task_labels[task]
            assert len(task_prob) == len(task_label)
            ap_random = get_ap_random(5, task_label)
            ap = average_precision_score(task_label, task_prob)
            if not np.isnan(ap):
                task_ap[task] = ap
                task_ap_random[task] = ap_random

        # compte class ap
        for obj_class in class_probs.keys():
            class_prob = class_probs[obj_class]
            class_label = class_labels[obj_class]
            merged_class_labels += class_labels[obj_class]
            assert len(class_prob) == len(class_label)
            ap_random = get_ap_random(5, class_label)
            ap = average_precision_score(class_label, class_prob)
            if not np.isnan(ap):
                class_ap[obj_class] = ap
                class_ap_random[obj_class] = ap_random

        # instance
        obj_ap = dict(obj_ap)
        obj_ap_random = dict(obj_ap_random)
  
        # task
        task_ap = dict(task_ap)
        task_ap_random = dict(task_ap_random)

        # class
        class_ap = dict(class_ap)
        class_ap_random = dict(class_ap_random)

        # merge 4 splits into 1, each instance/object/task will be held out once
        merged_task_ap = {**merged_task_ap, **task_ap}
        merged_class_ap = {**merged_class_ap, **class_ap}
        merged_obj_ap = {**merged_obj_ap, **obj_ap}

        merged_task_ap_random = {**merged_task_ap_random, **task_ap_random}
        merged_class_ap_random = {**merged_class_ap_random, **class_ap_random}
        merged_obj_ap_random = {**merged_obj_ap_random, **obj_ap_random}
    
    print(exp_name)

    # instance mAP
    obj_mAP = np.mean(list(merged_obj_ap.values()))  # 190 instances
    print(f"obj_mAP: {obj_mAP}")

    # class mAP
    class_mAP = np.mean(list(merged_class_ap.values()))
    print(f"class_mAP: {class_mAP}")
 
    # task mAP
    task_mAP = np.mean(list(merged_task_ap.values()))
    print(f"task_mAP: {task_mAP}")


