import argparse
import os
import copy
import pytorch_lightning as pl
import torch
from models.graspgpt_plain import GraspGPT_plain
from config import get_cfg_defaults

def get_timestamp():
    import datetime
    now = datetime.datetime.now()
    year = '{:02d}'.format(now.year)
    month = '{:02d}'.format(now.month)
    day = '{:02d}'.format(now.day)
    hour = '{:02d}'.format(now.hour)
    minute = '{:02d}'.format(now.minute)
    day_month_year = '{}-{}-{}-{}-{}'.format(year, month, day, hour, minute)
    return day_month_year


def main(cfg):
    
    model = GraspGPT_plain(cfg)

    if cfg.pretraining_mode == 1:
        weight_file = os.path.join(cfg.log_dir, cfg.pretrained_weight_file)
        if not os.path.exists(weight_file):
            raise FileNotFoundError(
                'Unable to find pre-trained pointnet file {}'.format(weight_file))

        pretrained_dict = torch.load(weight_file)['state_dict']
        model_dict = model.state_dict()

        # Copy over weights of PointNet Layers
        layers_updated = []
        for k in model_dict.keys():
            if k.find('SA_modules') >= 0 and (
                    k.find('bias') >= 0 or k.find('weight') >= 0):
                model_dict[k] = copy.deepcopy(pretrained_dict[k])
                layers_updated.append(k)

        model.load_state_dict(model_dict)

        print('Updated {} out of {} layers with pretrained weights'.format(
            len(layers_updated), len(list(model_dict.keys()))))

    exp_name = "{}_{}".format(cfg.name, get_timestamp())
    log_dir = os.path.join(cfg.log_dir, exp_name)

    early_stop_callback = pl.callbacks.EarlyStopping(patience=cfg.patience)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filepath=os.path.join(
            log_dir, 'weights', "best"
        ),
        verbose=True,
    )
    all_gpus = list(cfg.gpus)
    if len(all_gpus) == 1:
        torch.cuda.set_device(all_gpus[0])
    trainer = pl.Trainer(
        gpus=list(cfg.gpus),
        max_epochs=cfg.epochs,
        early_stop_callback=early_stop_callback,
        checkpoint_callback=checkpoint_callback,
        default_save_path=log_dir
    )
    trainer.fit(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN training")
    parser.add_argument(
        '--cfg_file',
        help='yaml file in YACS config format to override default configs',
        default='',
        type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    if args.cfg_file != '':
        if os.path.exists(args.cfg_file):
            cfg.merge_from_file(args.cfg_file)
        else:
            raise FileNotFoundError(args.cfg_file)

    if cfg.base_dir != '':
        if not os.path.exists(cfg.base_dir):
            raise FileNotFoundError(
                'Provided base dir {} not found'.format(
                    cfg.base_dir))
    else:
        assert cfg.base_dir == ''
        cfg.base_dir = os.path.join(os.path.dirname(__file__), '../data')

    if torch.cuda.is_available():
        print('Cuda is available, make sure you are training on GPU')

    if args.gpus == -1:
        args.gpus = [0, ]
    cfg.gpus = args.gpus

    if args.batch_size > -1:
        cfg.batch_size = args.batch_size

    cfg.freeze()
    print(cfg)
    main(cfg)
