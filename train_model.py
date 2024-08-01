import os
import io
import yaml
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import pdb
import sys
import torch.multiprocessing as mp
#import wandb
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from lib.datasets.datasets import AirplaneData
#from lib.datasets.cloud_transformations import ComposeCloudTransformation
from lib.networks.losses import Flow_Mixture_Loss
from lib.networks.flow_mixture import Flow_Mixture_Model
from lib.networks.optimizers import Adam, LRUpdater
from lib.networks.training import train, eval
from lib.networks.utils import cnt_params
from datetime import datetime
#from data_extract_labelled import dataset_airplanes
from evaluate_model import eval_main
from sinkhorn_pointcloud_alpha import visualize_data_with_SD_random
import pickle
import numpy as np
#from Data.data_gen.conditional_random_rotated_stars import conditional_data
#####
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#####

#input_clouds = conditional_data
gt_file = "/bigdata/hplsim/aipp/Gopal/data_npy/electron_data_processed.pkl"
with open(gt_file, 'rb') as file:
    input_clouds = pickle.load(file)

def define_options_parser():
    parser = argparse.ArgumentParser(description='Model training script. Provide a suitable config.')
    parser.add_argument('--config', type=str, default='/home/damoda95/Upload_v2/go_with_the_flows/configs/config_generative_modeling_FEL.yaml', help='Path to config file in YAML format.')
    parser.add_argument('--modelname', type=str, default = 'airplane_gen_model_v5', help='Model name for saving checkpoints.')
    parser.add_argument('--n_epochs', type=int, default = 10, help='Total number of training epochs.')
    parser.add_argument('--lr', type=float,default=0.000512, help='Learining rate value.')
    parser.add_argument('--cloud_random_rotate', action='store_true',
                        help='Flag signaling if we perform random 3D rotation during training.')
    parser.add_argument('--weights_type', type=str, default='global_weights',
                        help='choose to use global_weights/learned_weights.')
    parser.add_argument('--warmup_epoch', type=int, default=5, help='epochs use global_weights.')
    parser.add_argument('--jobid', type=str, default='1',
                        help='Id of training. If empty we give new id based of datetime.')
    parser.add_argument('--resume', action='store_true',
                        help='Flag signaling if training is resumed from a checkpoint.')
    parser.add_argument('--resume_optimizer', action='store_true',
                        help='Flag signaling if optimizer parameters are resumed from a checkpoint.')
    parser.add_argument('--distributed', type=bool, default=True, help='Flag if use distributed training')
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=4, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    return parser

def main_worker(gpu, ngpus_per_node, model_name, args):
    with io.open(args.config, 'r') as stream:
        config = yaml.full_load(stream)
    config['jobid'] = args.jobid
    if not 'logging_path' in config.keys():
        name_extension = config['jobid'] if config['jobid'] != '' else datetime.now().strftime("%Y%m%d_%H%M%S")
        config['logging_path'] = os.path.join(config['path2save'], args.modelname + '_' + name_extension)
        with open(args.config, 'w') as outfile:
            yaml.dump(config, outfile)
    if not os.path.exists(config['logging_path']) and gpu == 0:
        os.makedirs(config['logging_path'])
    config['model_name'] = "{}.pkl".format(model_name)
    config['n_epochs'] = args.n_epochs
    config['min_lr'] = config['max_lr'] = args.lr
    config['resume'] = True if args.resume else False
    config['resume_optimizer'] = True if args.resume_optimizer else False
    config['distributed'] = True if args.distributed else False
    config['logging'] = not args.distributed or (args.distributed and gpu == 0)
    config['cloud_random_rotate'] = args.cloud_random_rotate
    config['weights_type'] = args.weights_type
    print('Configurations loaded.', flush=True)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.world_size = args.gpus * args.nodes
        args.rank = args.nr * args.gpus + gpu
        torch.distributed.init_process_group(
            'nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
        print("world_size: ", args.world_size)
        print("rank: ", args.rank)

        config['batch_size'] = config['batch_size'] // args.world_size + int(
                            config['batch_size'] % args.world_size > gpu)
        print('Distributed training runs on GPU {} with batch size {}'.format(gpu, config['batch_size']))

    if not os.path.exists(os.path.join(config['logging_path'], 'config.yaml')) and gpu == 0:
        with open(os.path.join(config['logging_path'], 'config.yaml'), 'w') as outfile:
            yaml.dump(config, outfile)

    #cloud_transform, cloud_transform_val = ComposeCloudTransformation(**config)
    """
    train_dataset = ShapeNetCoreDataset(config['path2data'],
                                        part='train', meshes_fname=config['meshes_fname'],
                                        cloud_size=config['cloud_size'], return_eval_cloud=True,
                                        return_original_scale=config['cloud_rescale2orig'],
                                        cloud_transform=cloud_transform,
                                        chosen_label=config['chosen_label'])
    eval_dataset = ShapeNetCoreDataset(config['path2data'],
                                       part='val', meshes_fname=config['meshes_fname'],
                                       cloud_size=config['cloud_size'], return_eval_cloud=True,
                                       return_original_scale=config['cloud_rescale2orig'],
                                       cloud_transform=cloud_transform,
                                       chosen_label=config['chosen_label'])
    """
    
    train_dataset = AirplaneData(input_clouds)
    eval_dataset = AirplaneData(input_clouds)


    print('Dataset init: done.')
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, num_replicas=args.world_size, rank=args.rank)
        train_iterator = DataLoader(
            dataset=train_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True, drop_last=True, sampler=train_sampler)
        eval_iterator = DataLoader(
            eval_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True, drop_last=True, sampler=eval_sampler)
    else:
        train_iterator = DataLoader(
            dataset=train_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
            num_workers=config['num_workers'], pin_memory=True, drop_last=True)
        eval_iterator = DataLoader(
            eval_dataset, batch_size=config['batch_size'], shuffle=config['shuffle'],
            num_workers=config['num_workers'], pin_memory=True, drop_last=True)

    print(f'Size of training data: {len(train_dataset)}')
    print(f'Size of validation data: {len(eval_dataset)}')

    #wandb.init(project="electron_data_test", name="FEL_test_hemera")
    torch.cuda.set_device(gpu)
    model = Flow_Mixture_Model(**config)
    model = model.cuda(gpu)
    total_parameters = cnt_params(model.parameters())
    
    #wandb.log({"num_params": total_parameters})
    #wandb.log({"num_params_decoders": cnt_params(model.pc_decoder.parameters())})
    #wandb.log({"num_params_encoder": cnt_params(model.pc_encoder.parameters())})
    #wandb.log({"num_params_prior": cnt_params(model.g_prior.parameters())})
    
    print('Model init done on GPU {}.'.format(gpu))
    print('Total number of parameters: {} on GPU {}'.format(cnt_params(model.parameters()), gpu))
    print('Total number of parameters in decoder flows: {}'.format(cnt_params(model.pc_decoder.parameters())))
    print('Total number of parameters in Encoder: {}'.format(cnt_params(model.pc_encoder.parameters())))
    print('Total number of parameters in Prior network: {}'.format(cnt_params(model.g_prior.parameters())))
    print('Model init: done.')

    criterion = Flow_Mixture_Loss(**config).cuda(gpu)
    print('Loss init: done on GPU {}.'.format(gpu))

    optimizer = Adam(model.parameters(), lr=config['max_lr'], weight_decay=config['wd'],
                     betas=(config['beta1'], config['max_beta2']), amsgrad=True)

    scheduler = LRUpdater(len(train_iterator), **config)
    print('Optimizer init: done on GPU {}'.format(gpu))

    if not config['resume']:
        cur_epoch = 0
        cur_iter = 0
    else:
        path2checkpoint = os.path.join(config['logging_path'], config['model_name'])
        checkpoint = torch.load(path2checkpoint, map_location='cpu')
        cur_epoch = checkpoint['epoch']
        cur_iter = checkpoint['iter']
        model.load_state_dict(checkpoint['model_state'])
        if config['resume_optimizer']:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        del checkpoint
        print('Model {} loaded.'.format(path2checkpoint))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)

    print("training")
    # configure tensorboard logging
    #summary_writer = None
    #if gpu == 0:
        #tb_path = os.path.join(config['logging_path'], 'log')
        #summary_writer = SummaryWriter(tb_path)


    #min_loss = 10000
    for epoch in range(cur_epoch, config['n_epochs']):
        warmup = False
        train(train_iterator, model, criterion, optimizer, scheduler, epoch, cur_iter, warmup, **config)
        #min_loss = eval(eval_iterator, model, criterion, optimizer, epoch, cur_iter, warmup, min_loss, summary_writer, **config)
        cur_iter = 0

    #summary_writer.close()

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def train_main(model_name):
    parser = define_options_parser()
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed and ngpus_per_node > 1:
        #breakpoint()
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = find_free_port()  # '6666'
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, model_name, args))
    else:
        args.distributed = False
        main_worker(0, 1, model_name, args)

def train_gen():
    cond_type = 'continuous'
    model_name = 'electron_data_model_test_2'
    train_main(model_name)

    for cond_fel_line in range(8):
        cond = torch.tensor(np.array([18.1, 12.15, float(cond_fel_line), 1.467, 1.933]))
        generated_file = f"{model_name}_{cond_type}_{str(cond)}.h5"
        eval_main(model_name, generated_file, cond)
        visualize_data_with_SD_random(generated_file, cond) 
        
if __name__ == '__main__':
    train_gen()
