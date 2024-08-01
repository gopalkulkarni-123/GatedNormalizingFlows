import os
from time import time
from sys import stdout

import h5py as h5
import numpy as np
import torch

from lib.networks.utils import AverageMeter
from lib.networks.utils import JSD, f_score
#from lib.metrics.evaluation_metrics import compute_all_metrics, EMD_CD_F1, distChamferCUDA, emd_approx

def one_hot_vector(index, size):
    """
    Generate a one-hot vector of a given size with the specified index set to 1.
    
    Parameters:
    - index (int): The index to set to 1.
    - size (int): The size of the one-hot vector.
    
    Returns:
    - numpy.ndarray: The one-hot vector.
    """
    if index < 0 or index >= size:
        raise ValueError("Index must be within the range [0, size).")

    one_hot = np.zeros(size)
    one_hot[int(index)] = 1

    return one_hot

def group_tensors(A, B, number_flows, dims):
    A=A.cpu()
    B=B.cpu()
    # Expand dimensions of tensor B to match the dimensions of tensor A
    B = B.unsqueeze(1)
    B = B.expand(-1, A.size(1), -1)
    
    grouped_tensors = []
    for label in range(number_flows):
        C = torch.zeros((0, dims))
        for value in range (A.shape[2]):
            if B[0][0][value] == label+1:
                C = torch.cat((C, A[:, :, value]), dim=0)
        grouped_tensors.append(C.T)
    return grouped_tensors


def save_clouds(complete_cloud_tensor, partwise_cloud_list, file_name, log_weights):
    file = h5.File('/home/damoda95/Upload/results/airplane_gen_model/{}'.format(file_name), 'w')
    group_summed = file.create_group('all_tensors')
    group_summed.create_dataset('summed_tensor', data=complete_cloud_tensor)
    for i in range(len(partwise_cloud_list)):
        group_summed.create_dataset('decoder_part_{}'.format(i), data=partwise_cloud_list[i])
    group_summed.create_dataset('log_weights', data = log_weights.cpu())
    file.close()

def evaluate(iterator, model, generated_file, loss_func, cond, **kwargs):
    train_mode = kwargs.get('train_mode')
    util_mode = kwargs.get('util_mode')
    cond_type = kwargs.get('cond_type')
    #is_saving = kwargs.get('saving')

    batch_time = AverageMeter()
    data_time = AverageMeter()
    inf_time = AverageMeter()

    #gen_clouds_buf = []
    #ref_clouds_buf = []

    model.eval()
    torch.set_grad_enabled(False)

    end = time()

    
    data_time.update(time() - end)

    n_components = kwargs.get('n_components')
    n_points = kwargs.get('sampled_cloud_size')
    dims = kwargs.get('p_latent_space_size')
    print("dims is", dims)

    g_clouds = torch.zeros((1,dims,n_points)).cuda(non_blocking=True)
    p_clouds = torch.zeros((1,dims,n_points)).cuda(non_blocking=True).cuda(non_blocking=True)
    if cond_type == 'continuous':
        cloud_labels = cond.clone().detach().unsqueeze(0).float().cuda(non_blocking=True)
        #cloud_labels = torch.tensor(cond).unsqueeze(0).float().cuda(non_blocking=True)
        #cloud_labels = torch.tensor(np_label).unsqueeze(0).float().cuda(non_blocking=True)        
        #cloud_labels = torch.tensor([cond_num]).unsqueeze(1).float().cuda(non_blocking=True)
    if cond_type == 'discrete':
        cloud_labels = torch.tensor(one_hot_vector(cond, 4)).unsqueeze(0).float().cuda(non_blocking=True)
    #cloud_labels = torch.tensor(one_hot_vector(0, 4)).cuda(non_blocking=True)
    inf_end = time()

    # for test, generate samples
    with torch.no_grad():
        if train_mode == 'p_rnvp_mc_g_rnvp_vae':
            output_prior, samples, labels, log_weights = model(g_clouds, p_clouds, cloud_labels, images=None, n_sampled_points=n_points, labeled_samples=True)
    #k = log_weights
    inf_time.update((time() - inf_end) / g_clouds.shape[0], g_clouds.shape[0])
    samples = samples.cpu()
    summed_clouds = samples
    r_clouds = group_tensors(samples, labels, n_components, dims)
    save_clouds(summed_clouds, r_clouds, generated_file, log_weights)

    batch_time.update(time() - end)
    end = time()

    print('Inference time: {} sec/sample'.format(inf_time.avg))

    res = {}
    return res