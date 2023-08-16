import argparse
import os
import copy
import pickle
import sys
import time
import numpy as np
import torch
import torch.utils.data as data
import tqdm

BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, '../../'))
from visualize import draw_scene, get_gripper_control_points
from geometry_utils import regularize_pc_point_count
from data.SGNLoader import pc_normalize, get_task1_hits
from data.data_specification import TASKS
from utils.splits import get_split_data, parse_line, get_ot_pairs_taskgrasp


class GCNTaskGrasp(data.Dataset):
    def __init__(
            self,
            num_points,
            transforms=None,
            train=0,
            download=True,
            base_dir=None,
            folder_dir='',
            normal=True,
            tasks=None,
            map_obj2class=None,
            class_list=None,
            split_mode=None,
            split_idx=0,
            split_version=0,
            pc_scaling=True,
            use_task1_grasps=True,
            graph_data_path='',
            include_reverse_relations=True,
            subgraph_sampling=True,
            sampling_radius=2,
            instance_agnostic_mode=1):
        """

        Args:
            num_points: Number of points in point cloud (used to downsample data to a fixed number)
            transforms: Used for data augmentation during training
            train: 1 for train, 0 for test, 2 for validation
            base_dir: location of dataset
            folder_dir: name of dataset
            tasks: list of tasks
            class_list: list of object classes
            map_obj2class: dictionary mapping dataset object to corresponding WordNet class
            split_mode: Choose between held-out instance ('i'), tasks ('t') and classes ('o')
            split_version: Choose 1, 0 is deprecated
            split_idx: For each split mode, there are 4 cross validation splits (choose between 0-3)
            pc_scaling: True if you want to scale the point cloud by the standard deviation
            include_reverse_relations: True since we are modelling a undirected graph
            use_task1_grasps: True if you want to include the grasps from the object-task pairs
                rejected in Stage 1 (and add these grasps are negative samples)

            Deprecated args (not used anymore): normal, download, 
                instance_agnostic_mode, subgraph_sampling, sampling_radius
        """
        super().__init__()
        if graph_data_path != '':
            graph_data_path = os.path.join(
                base_dir,
                'knowledge_graph',
                graph_data_path,
                'graph_data.pkl')  # load pre-constructed graph: 'gcngrasp/../data/knowledge_graph/kb2_task_wn_noi/graph_data.pkl'
            assert os.path.exists(graph_data_path)
        self._graph_data_path = graph_data_path
        self._pc_scaling = pc_scaling  # True
        self._split_mode = split_mode  # t/o
        self._split_idx = split_idx  # 0,1,2,3
        self._split_version = split_version  # always '1' for now
        self._num_points = num_points  # downsample to 4096
        self._transforms = transforms  # no transform for eval
        self._tasks = tasks  # 56 tasks
        self._num_tasks = len(self._tasks)

        task1_results_file = os.path.join(
            base_dir, folder_dir, 'task1_results.txt')
        assert os.path.exists(task1_results_file)

        self._train = train
        self._map_obj2class = map_obj2class  # instance to class
        data_dir = os.path.join(base_dir, folder_dir, "scans")
        self.obj_gpt_dir = os.path.join(base_dir, folder_dir, "obj_gpt_v2")
        self.task_gpt_dir = os.path.join(base_dir, folder_dir, 'task_gpt_v2')
        self.task_ins_dir = os.path.join(base_dir, folder_dir, "task_ins_v2")

        data_txt_splits = {
            0: 'test_split.txt',
            1: 'train_split.txt',
            2: 'val_split.txt'}

        if self._train not in data_txt_splits:
            raise ValueError("Unknown split arg {}".format(self._train))

        self._parse_func = parse_line
        lines = get_split_data(
            base_dir,
            folder_dir,
            self._train,
            self._split_mode,
            self._split_idx,
            self._split_version,
            use_task1_grasps,  # True for eval
            data_txt_splits,
            self._map_obj2class,
            self._parse_func,
            get_ot_pairs_taskgrasp,
            get_task1_hits)

        self._data = []
        self._pc = {}
        self._grasps = {}
        self._object_classes = class_list

        self._num_object_classes = len(self._object_classes)

        start = time.time()
        correct_counter = 0

        all_object_instances = []

        self._object_task_pairs_dataset = []
        self._data_labels = []
        self._data_label_counter = {0: 0, 1: 0}  # record correct and wrong grasps

        # 190 instances, 25 grasps per instance, 56 tasks per grasp, 190*25*56 / 4 = 66500
        for i in tqdm.trange(len(lines)):

            obj, obj_class, grasp_id, task, label = parse_line(lines[i])  # e.g. 039_brush, brush, 8, till, False
            obj_class = self._map_obj2class[obj]  # 039_brush -> 'scrub_brush.n.01'
            all_object_instances.append(obj)
            self._object_task_pairs_dataset.append("{}-{}".format(obj, task))

            # load point cloud
            pc_file = os.path.join(data_dir, obj, "fused_pc_clean.npy")
            if pc_file not in self._pc:
                if not os.path.exists(pc_file):
                    raise ValueError(
                        'Unable to find processed point cloud file {}'.format(pc_file))
                pc = np.load(pc_file)  # [49093, 6] xyz, rgb
                pc_mean = pc[:, :3].mean(axis=0)
                pc[:, :3] -= pc_mean
                self._pc[pc_file] = pc  # e.g. {'gcngrasp/../data/tas..._clean.npy': array([[ 2.30503132e...000e+02]])}

            # load grasps
            grasp_file = os.path.join(
                data_dir, obj, "grasps", str(grasp_id), "grasp.npy")
            if grasp_file not in self._grasps:
                grasp = np.load(grasp_file)
                self._grasps[grasp_file] = grasp

            self._data.append(
                (grasp_file, pc_file, obj, obj_class, grasp_id, task, label))
            self._data_labels.append(int(label))
            if label:
                correct_counter += 1
                self._data_label_counter[1] += 1
            else:
                self._data_label_counter[0] += 1
            
        self._all_object_instances = list(set(all_object_instances))
        self._len = len(self._data)
        print('Loading files from {} took {}s; overall dataset size {}, proportion successful grasps {:.2f}'.format(
            data_txt_splits[self._train], time.time() - start, self._len, float(correct_counter / self._len)))

        self._data_labels = np.array(self._data_labels)  # 66500

    @property
    def weights(self):
        N = self.__len__()
        weights = {
                0: float(N) /
                self._data_label_counter[0],
                1: float(N) /
                self._data_label_counter[1]}
        weights_sum = sum(weights.values())
        weights[0] = weights[0] / weights_sum
        weights[1] = weights[1] / weights_sum
        weights_data = np.zeros(N)
        weights_data[self._data_labels == 0] = weights[0]
        weights_data[self._data_labels == 1] = weights[1]
        return weights_data

    def __getitem__(self, idx):
        
        grasp_file, pc_file, obj, obj_class, grasp_id, task, label = self._data[idx]
        pc = self._pc[pc_file]  # [49093, 6]
        pc = regularize_pc_point_count(
            pc, self._num_points, use_farthest_point=False)  # [4096, 6]
        pc_color = pc[:, 3:]  # [4096, 3]
        pc = pc[:, :3]  # [4096, 3]

        obj_class_ = obj_class.split('.')[0]  # scrub_brush.n.01 -> sruch_brush
        obj_desc_dir =  os.path.join(self.obj_gpt_dir, obj_class_, 'descriptions', str(np.random.randint(0, 10)))
        if not os.path.exists(obj_desc_dir):
            raise ValueError(f"No such object description dir: {obj_desc_dir}")
        obj_desc = np.load(os.path.join(obj_desc_dir, 'word_embed.npy'))[0]
        obj_desc_mask = np.load(os.path.join(obj_desc_dir, 'attn_mask.npy'))[0]

        task_desc_dir = os.path.join(self.task_gpt_dir, task, 'descriptions', str(np.random.randint(0, 10)))
        if not os.path.exists(task_desc_dir):
            raise ValueError(f"No such task description dir: {task_desc_dir}")
        task_desc = np.load(os.path.join(task_desc_dir, 'word_embed.npy'))[0]
        task_desc_mask = np.load(os.path.join(task_desc_dir, 'attn_mask.npy'))[0]

        task_ins_id = np.random.randint(0, 53)
        task_ins_path = os.path.join(self.task_ins_dir, obj_class_, task, str(task_ins_id)+'_word.npy')
        task_ins_mask_path = os.path.join(self.task_ins_dir, obj_class_, task, str(task_ins_id)+'_mask.npy')
        if not os.path.exists(task_ins_path) or not os.path.exists(task_ins_mask_path):
            raise ValueError(f"No such task instruction or mask file: {task_ins_path}")
        with open(task_ins_path, 'rb') as f:
            task_ins = np.load(f)[0]  # [21, 768]
        with open(task_ins_mask_path, 'rb') as f:
            task_ins_mask = np.load(f)[0]  # [21]

        grasp = self._grasps[grasp_file]  # [4, 4]
        task_id = self._tasks.index(task)  # int  
        class_id = self._object_classes.index(obj_class)  # int8
        instance_id = self._all_object_instances.index(obj)  # int8

        grasp_pc = get_gripper_control_points()  # [7, 4]
        grasp_pc = np.matmul(grasp, grasp_pc.T).T
        grasp_pc = grasp_pc[:, :3]  # [7, 3]

        latent = np.concatenate(
            [np.zeros(pc.shape[0]), np.ones(grasp_pc.shape[0])])
        latent = np.expand_dims(latent, axis=1)
        pc = np.concatenate([pc, grasp_pc], axis=0)  # [4103, 3]

        pc, grasp = pc_normalize(pc, grasp, pc_scaling=self._pc_scaling)  # [4103, 3], [4, 4]
        pc = np.concatenate([pc, latent], axis=1)  # [4103, 4]

        if self._transforms is not None:
            pc = self._transforms(pc)

        label = float(label)
       
        return pc, pc_color, task_id, class_id, instance_id, grasp, label, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask


    def __len__(self):
        return self._len

    @staticmethod
    def collate_fn(batch):
        """ This function overrides defaul batch collate function and aggregates 
        the graph and point clound data across the batch into a single graph tensor """

        pc = torch.stack([torch.as_tensor(_[0]) for _ in batch], dim=0)
        pc_color = torch.stack([torch.as_tensor(_[1]) for _ in batch], dim=0)
        task_id = torch.stack([torch.tensor(_[2]) for _ in batch], dim=0)
        # task_gid = torch.stack([torch.tensor(_[3]) for _ in batch], dim=0)
        # instance_gid = torch.stack([torch.tensor(_[4]) for _ in batch], dim=0)
        # obj_class_gid = torch.stack([torch.tensor(_[5]) for _ in batch], dim=0)
        class_id = torch.stack([torch.tensor(_[3]) for _ in batch], dim=0)
        instance_id = torch.stack([torch.tensor(_[4]) for _ in batch], dim=0)
        grasp = torch.stack([torch.tensor(_[5]) for _ in batch], dim=0)
        # node_x_idx = torch.cat([torch.tensor(_[9]) for _ in batch], dim=0)
        label = torch.stack([torch.tensor(_[6]) for _ in batch], dim=0)

        obj_desc = torch.stack([torch.tensor(_[7]) for _ in batch], dim=0)
        obj_desc_mask = torch.stack([torch.tensor(_[8]) for _ in batch], dim=0)

        task_desc = torch.stack([torch.tensor(_[9]) for _ in batch], dim=0)
        task_desc_mask = torch.stack([torch.tensor(_[10]) for _ in batch], dim=0)

        task_ins = torch.stack([torch.tensor(_[11]) for _ in batch], dim=0)
        task_ins_mask = torch.stack([torch.tensor(_[12]) for _ in batch], dim=0)

        return pc, pc_color, task_id, class_id, instance_id, grasp, label, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument('--base_dir', default='', type=str)
    args = parser.parse_args()

    if args.base_dir != '':
        if not os.path.exists(args.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    args.base_dir))
    else:
        assert args.base_dir == ''
        args.base_dir = os.path.join(os.path.dirname(__file__), '../../data')

    folder_dir = 'taskgrasp'
    _, _, _, name2wn = pickle.load(
        open(os.path.join(base_dir, folder_dir, 'misc.pkl'), 'rb'))

    dset = GCNTaskGrasp(
        4096,
        transforms=None,
        train=2,  # validation
        base_dir=base_dir,
        folder_dir=folder_dir,
        normal=False,
        tasks=TASKS,
        map_obj2class=name2wn,
        split_mode='t',
        split_idx=0,
        split_version=0,
        pc_scaling=True,
        use_task1_grasps=True,
        graph_data_path='kb2_task_wn_noi',
        include_reverse_relations=True,
        subgraph_sampling=True,
        sampling_radius=2,
        instance_agnostic_mode=1
    )

    weights = dset.weights
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights, len(weights))

    dloader = torch.utils.data.DataLoader(
        dset,
        batch_size=16,
        sampler=sampler,
        collate_fn=GraspGCNDataset.collate_fn)

    with torch.no_grad():
        for batch in dloader:
            pc, pc_color, task_id, task_gid, instance_gid, obj_class_gid, class_id, instance_id, grasp, node_x_idx, latent, edge_index, label = batch
