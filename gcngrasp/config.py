from yacs.config import CfgNode as CN

# Miscellaneous configs
_C = CN()
_C.weight_file = ''
_C.batch_size = 32
_C.num_points = 4096
_C.epochs = 50
_C.gpus = [0, ]
_C.distrib_backend = 'dp'
_C.name = 'gcngrasp'
_C.algorithm_class = 'GCNGrasp'
_C.dataset_class = 'GCNTaskGrasp'
_C.base_dir = ''
_C.folder_dir = 'taskgrasp'
_C.log_dir = 'checkpoints/'
_C.split_mode = 'o'
_C.embedding_size = 256
_C.split_idx = 0
_C.split_version = 1  # 0 is for random splits, 1 is for cross-validation splits
_C.patience = 50
_C.pc_scaling = True
_C.use_task1_grasps = True
_C.weighted_sampling = True
_C.use_class_list = True
_C.pretraining_mode = 0  # This is just for pointnet layers. 0 for no pretraining, 1 for loading pretrained weights
_C.pretrained_weight_file = ''
_C.embedding_mode = 0 # 0 for random initialization, 1 for finetuning pretrained weights, 2 for loading and freezing pretrained weights
_C.embedding_model = 'numberbatch'

# Graph GCN configs
_C.graph_data_path = 'kb2_task_wn_noi'
_C.include_reverse_relations = True
_C.gcn_num_layers = 6
_C.gcn_conv_type = 'GCNConv'
_C.subgraph_sampling = True # DEPRECATED
_C.sampling_radius = 2 # DEPRECATED
_C.gcn_skip_mode = 0 # DEPRECATED
_C.instance_agnostic_mode = 1 # DEPRECATED

# Model and training configs
_C.model = CN()
_C.model.use_xyz = True
_C.model.use_normal = False

_C.optimizer = CN()
_C.optimizer.lr_decay = 0.7
_C.optimizer.lr = 1e-4
_C.optimizer.decay_step = 2e4
_C.optimizer.bn_momentum = 0.5
_C.optimizer.bnm_decay =0.5
_C.optimizer.weight_decay = 0.0001
_C.optimizer.lr_clip = 1e-5
_C.optimizer.bnm_clip = 1e-2

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
