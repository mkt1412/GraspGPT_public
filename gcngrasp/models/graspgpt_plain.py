import sys
import os
import pickle
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
from torch.utils.data import DataLoader
from torchvision import transforms
from data.GCNLoader import GCNTaskGrasp
from data.data_specification import TASKS
import data.data_utils as d_utils


class GraspGPT_plain(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self._build_model()
    
    def _build_model(self):

        pc_dim = 1

        # pointNet set abstraction
        self.SA_modules = nn.ModuleList()
        
        # groupers -> MLPs
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[pc_dim, 32, 32, 64], [pc_dim, 64, 64, 128], [pc_dim, 64, 96, 128]],
                use_xyz=self.cfg.model.use_xyz,
            )
        )

        input_channels = 64 + 128 + 128

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=self.cfg.model.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[128 + 256 + 256, 256, 512, 1024],
                use_xyz=self.cfg.model.use_xyz,
            )
        )

        # language
        self.ins_preprocess = nn.Linear(768, 128)
        self.task_preprocess = nn.Linear(768, 128)
        self.obj_preprocess = nn.Linear(768, 128)

        _, _, _, self.name2wn = pickle.load(open(os.path.join(self.cfg.base_dir, self.cfg.folder_dir, 'misc.pkl'),'rb'))
        self._class_list = pickle.load(open(os.path.join(self.cfg.base_dir, 'class_list.pkl'),'rb')) if self.cfg.use_class_list else list(self.name2wn.values())

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, self.cfg.embedding_size)
        )

        self.fc_layer3 = nn.Sequential(
            nn.Linear(428+128+128, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features
    
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings / sum_mask
    
    def forward(self, pointcloud, obj_desc, obj_desc_mask, task_desc, task_desc_mask, ins, ins_mask):

        # pointcloud: [32, 4103, 4]
        xyz, features = self._break_up_pc(pointcloud)  # [32, 4103, 3], [32, 1, 4103] 4103 = 4096(object) + 7(gripper)

        # pointnetSAMSG * 2 + pointSA
        for i, module in enumerate(self.SA_modules):
            xyz, features = module(xyz, features)
            # print(xyz.shape, features.shape)
            # torch.Size([32, 512, 3]) torch.Size([32, 320, 512])
            # torch.Size([32, 128, 3]) torch.Size([32, 640, 128])
            # None                     torch.Size([32, 1024, 1])
        shape_embedding = self.fc_layer(features.squeeze(-1))  # [32, 300], object+gripper

        ins_embed = self.ins_preprocess(ins)  # [B, 21, 128]
        task_embed = self.task_preprocess(task_desc)  # [B, 180, 128]
        obj_mebed = self.obj_preprocess(obj_desc)

        ins_embedding = self.mean_pooling(ins_embed, ins_mask)  # [B, 128]
        task_embedding = self.mean_pooling(task_embed, task_desc_mask)  # [B, 128]
        obj_embedding = self.mean_pooling(obj_mebed, obj_desc_mask)  # [B, 128]

        embedding_in = torch.concat([shape_embedding, ins_embedding, task_embedding, obj_embedding], dim=-1)  # [B, 300+128+128+128]  

        logits = self.fc_layer3(embedding_in)

        return logits

    def training_step(self, batch, batch_idx):
        pc, _, task_id, class_id, _, _, label, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask = batch

        # logits = self.forward(pc, node_x_idx, latent, edge_index, tasks, tasks_no, classes, classes_no)
        logits = self.forward(pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
        logits = logits.squeeze()

        # bce loss
        loss = F.binary_cross_entropy_with_logits(logits, label.type(torch.cuda.FloatTensor))
        
        with torch.no_grad():
            pred = torch.round(torch.sigmoid(logits))
            acc = (pred == label).float().mean()

        log = dict(train_loss=loss, train_acc=acc)

        return dict(loss=loss, log=log, progress_bar=dict(train_acc=acc))

    def validation_step(self, batch, batch_idx):

        pc, _, task_id, class_id, _, _, label, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask = batch

        logits = self.forward(pc, obj_desc, obj_desc_mask, task_desc, task_desc_mask, task_ins, task_ins_mask)
        logits = logits.squeeze()

        try:
            loss = F.binary_cross_entropy_with_logits(logits, label.type(torch.cuda.FloatTensor))
        except ValueError:
            assert label.type(torch.cuda.FloatTensor).shape[0] == 1
            logits = logits.unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, label.type(torch.cuda.FloatTensor))

        pred = torch.round(torch.sigmoid(logits))
        acc = (pred == label).float().mean()

        return dict(val_loss=loss, val_acc=acc)

    def validation_end(self, outputs):
        reduced_outputs = {}
        # process val_loss and val_acc
        for k in outputs[0]:
            for o in outputs:
                reduced_outputs[k] = reduced_outputs.get(k, []) + [o[k]]

        # average over all outputs
        for k in reduced_outputs:
            reduced_outputs[k] = torch.stack(reduced_outputs[k]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs
    
    def configure_optimizers(self):
        lr_lbmd = lambda _: max(
            self.cfg.optimizer.lr_decay
            ** (
                int(
                    self.global_step
                    * self.cfg.batch_size
                    / self.cfg.optimizer.decay_step
                )
            ),
            self.cfg.optimizer.lr_clip / self.cfg.optimizer.lr,
        )
        # bn_lbmd = lambda _: max(
        #     self.cfg.optimizer.bn_momentum
        #     * self.cfg.optimizer.bnm_decay
        #     ** (
        #         int(
        #             self.global_step
        #             * self.cfg.batch_size
        #             / self.cfg.optimizer.decay_step
        #         )
        #     ),
        #     self.cfg.optimizer.bnm_clip,
        # )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd)
        # bnm_scheduler = BNMomentumScheduler(self, bn_lambda=bn_lbmd)

        return [optimizer], [lr_scheduler]
    
    def prepare_data(self):
        """ Initializes datasets used for training, validation and testing """

        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudGraspToTensor(),
                d_utils.PointcloudGraspScale(),
                d_utils.PointcloudGraspRotate(axis=np.array([1.0, 0.0, 0.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspRotate(axis=np.array([0.0, 1.0, 0.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspRotate(axis=np.array([0.0, 0.0, 1.0])),
                d_utils.PointcloudGraspRotatePerturbation(),
                d_utils.PointcloudGraspTranslate(),
                d_utils.PointcloudGraspJitter(),
                d_utils.PointcloudGraspRandomInputDropout(),
            ]
        )

        self.train_dset = GCNTaskGrasp(
            self.cfg.num_points,
            transforms=train_transforms,
            train=1,
            base_dir=self.cfg.base_dir,
            folder_dir=self.cfg.folder_dir,
            normal=self.cfg.model.use_normal,
            tasks=TASKS,
            map_obj2class=self.name2wn,
            class_list=self._class_list,
            split_mode=self.cfg.split_mode,
            split_idx=self.cfg.split_idx,
            split_version=self.cfg.split_version,
            pc_scaling=self.cfg.pc_scaling,
            use_task1_grasps=self.cfg.use_task1_grasps,
            graph_data_path=self.cfg.graph_data_path,
            include_reverse_relations=self.cfg.include_reverse_relations,
            subgraph_sampling=self.cfg.subgraph_sampling,
            sampling_radius=self.cfg.sampling_radius,
            instance_agnostic_mode=self.cfg.instance_agnostic_mode
        )

        if self.cfg.weighted_sampling:
            weights = self.train_dset.weights
            self._train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        self.val_dset = GCNTaskGrasp(
            self.cfg.num_points,
            transforms=train_transforms,
            train=2,
            base_dir=self.cfg.base_dir,
            folder_dir=self.cfg.folder_dir,
            normal=self.cfg.model.use_normal,
            tasks=TASKS,
            map_obj2class=self.name2wn,
            class_list=self._class_list,
            split_mode=self.cfg.split_mode,
            split_idx=self.cfg.split_idx,
            split_version=self.cfg.split_version,
            pc_scaling=self.cfg.pc_scaling,
            use_task1_grasps=self.cfg.use_task1_grasps,
            graph_data_path=self.cfg.graph_data_path,
            include_reverse_relations=self.cfg.include_reverse_relations,
            subgraph_sampling=self.cfg.subgraph_sampling,
            sampling_radius=self.cfg.sampling_radius,
            instance_agnostic_mode=self.cfg.instance_agnostic_mode
        )

    def _build_dataloader(self, dset, mode):
        if self.cfg.weighted_sampling and mode == "train":
            return DataLoader(
                dset,
                batch_size=self.cfg.batch_size,
                num_workers=4,
                pin_memory=True,
                drop_last=mode == "train",
                sampler=self._train_sampler,
                collate_fn=GCNTaskGrasp.collate_fn
            )
        else:
            return DataLoader(
                dset,
                batch_size=self.cfg.batch_size,
                shuffle=mode == "train",
                num_workers=4,
                pin_memory=True,
                drop_last=mode == "train",
                collate_fn=GCNTaskGrasp.collate_fn
            )

    def train_dataloader(self):
        return self._build_dataloader(self.train_dset, mode="train")

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
