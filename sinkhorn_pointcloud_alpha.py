import h5py
import torch
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import yaml
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.autograd import Variable
#import wandb
import itertools
#import pdb

def overlay_scatter_plots(tensor1, tensor2, title, color1, color2, file_name_save):
    """
    Generate scatter plots overlaying two sets of points.

    Parameters:
    - tensor1: numpy array or tensor of shape (n, d) representing the first set of points.
    - tensor2: numpy array or tensor of shape (m, d) representing the second set of points.
    - title: Title for the plot.
    - color1: Color for points in tensor1.
    - color2: Color for points in tensor2.
    - file_name_save: File name to save the plot.

    Returns:
    - None
    """
    d1, n = tensor1.shape
    d2, m = tensor2.shape
    
    assert d1 == d2, "The dimensions of the two tensors must be the same."
    
    # Generate all possible 2D combinations of the dimensions
    combinations = list(itertools.combinations(range(d1), 2))

    # Create subplots for each combination
    num_plots = len(combinations)
    num_cols = 3  # You can adjust the number of columns as per your preference
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle(title)

    # Flatten axes if necessary
    axes = axes.flatten()

    # Plot each combination for tensor1
    for i, (dim1, dim2) in enumerate(combinations):
        ax = axes[i]
        ax.scatter(tensor1[dim1], tensor1[dim2], marker='o', color=color1, label='Ground Truth', s=50)
        ax.scatter(tensor2[dim1], tensor2[dim2], marker='o', color=color2, label='Generated Data', s=50)
        ax.set_xlabel(f'Dimension {dim1 + 1}')
        ax.set_ylabel(f'Dimension {dim2 + 1}')
        ax.set_title(f'Scatter Plot: Dimension {dim1 + 1} vs Dimension {dim2 + 1}')
        #ax.set_xlim([-0.1, 1.1])
        #ax.set_ylim([-0.1, 1.1])
        ax.legend()

    # Hide empty subplots, if any
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
    #plt.savefig(r"D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\Thesis\comparison_analysis\Report_9_02\Angled_gt\sinkhorn_value_images\{}.png".format(file_name_save))
    #plt.show()
    #wandb.log({title: wandb.Image(plt)})
    plt.close()

def scatter_plots(tensor, title, color, file_name_save, axis_lims):
    """
    Generate scatter plots for all possible 2D combinations of the dimensions in the input tensor.

    Parameters:
    - tensor: numpy array or tensor of shape (n, d) where d is the number of dimensions.

    Returns:
    - None
    """
    d, n = tensor.shape
    #print(f"d is {d} and n is {n}")

    # Generate all possible 2D combinations of the dimensions
    combinations = list(itertools.combinations(range(d), 2))

    # Create subplots for each combination
    num_plots = len(combinations)
    num_cols = 3  # You can adjust the number of columns as per your preference
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle(title)

    # Flatten axes if necessary
    axes = axes.flatten()

    # Plot each combination
    for i, (dim1, dim2) in enumerate(combinations):
        ax = axes[i]
        ax.scatter(tensor[dim1], tensor[dim2], marker='o',color = color, s=50)
        ax.set_xlabel(f'Dimension {dim1 + 1}')
        ax.set_ylabel(f'Dimension {dim2 + 1}')
        ax.set_title(f'Scatter Plot: Dimension {dim1 + 1} vs Dimension {dim2 + 1} ')


        x_min = float(axis_lims[0][dim1]) + .005
        x_max = float(axis_lims[1][dim1]) + .005
        y_min = float(axis_lims[0][dim2]) + .005
        y_max = float(axis_lims[1][dim2]) + .005
        
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

    # Hide empty subplots, if any
    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
    plt.savefig("/home/damoda95/Upload/results/airplane_gen_model/{}.png".format(file_name_save))
    #wandb.log({title: wandb.Image(plt)})    
    plt.close()

def sinkhorn_normalized(x, y, epsilon, n, niter):

    Wxy = sinkhorn_loss(x, y, epsilon, n, niter)
    Wxx = sinkhorn_loss(x, x, epsilon, n, niter)
    Wyy = sinkhorn_loss(y, y, epsilon, n, niter)
    return 2 * Wxy - Wxx - Wyy

def sinkhorn_loss(x, y, epsilon, n, niter):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop
    """

    # The Sinkhorn algorithm takes as input three variables :
    C = Variable(cost_matrix(x, y))  # Wasserstein cost function

    # both marginals are fixed with equal weights
    # mu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    # nu = Variable(1. / n * torch.cuda.FloatTensor(n).fill_(1), requires_grad=False)
    mu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)
    nu = Variable(1. / n * torch.FloatTensor(n).fill_(1), requires_grad=False)

    # Parameters of the Sinkhorn algorithm.
    rho = 1  # (.5) **2          # unbalanced transport
    tau = -.8  # nesterov-like acceleration
    lam = rho / (rho + epsilon)  # Update exponent
    thresh = 10**(-1)  # stopping criterion

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(1) + v.unsqueeze(0)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = 0. * mu, 0. * nu, 0.
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
        v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        # accelerated unbalanced iterations
        # u = ave( u, lam * ( epsilon * ( torch.log(mu) - lse(M(u,v)).squeeze()   ) + u ) )
        # v = ave( v, lam * ( epsilon * ( torch.log(nu) - lse(M(u,v).t()).squeeze() ) + v ) )
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).data.numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost

def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def scatter_plot_2d_combinations(point_cloud, title, color_arg, file_name_save):
    """
    Generate scatter plots for all possible 2D combinations of a 4D point cloud.

    Parameters:
    - point_cloud (numpy array): 4D point cloud with shape (num_points, 4).

    Returns:
    - None (displays scatter plots).
    """
    num_dimensions = point_cloud.shape[0]

    # Generate all 2D combinations
    combinations_2d = list(combinations(range(num_dimensions), 2))

    # Create subplots based on the number of combinations
    num_plots = len(combinations_2d)
    num_rows = int(np.ceil(np.sqrt(num_plots)))
    num_cols = int(np.ceil(num_plots / num_rows))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    fig.suptitle(title)

    for i, combination in enumerate(combinations_2d):
        row = i // num_cols
        col = i % num_cols

        x_index, y_index = combination
        x_values = point_cloud[x_index]
        y_values = point_cloud[y_index]

        if num_rows > 1:
            # If more than one row, use subplots
            axes[row, col].scatter(x_values, y_values, s=10, alpha=0.5, color= color_arg)
            axes[row, col].set_title(f"Dimensions {x_index+1} vs {y_index+1}")
            axes[row, col].set_xlim(-7, 7)
            axes[row, col].set_ylim(-7, 7)
        else:
            # If only one row, use a single subplot
            axes[col].scatter(x_values, y_values, s=10, alpha=0.5)
            axes[col].set_title(f"Dimensions {x_index+1} vs {y_index+1}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout
    #wandb.log({"image": wandb.Image(plt)})
    #plt.savefig("/home/goku454c/experiments_train_gen_conditional_prior/results/airplane_gen_model/six_dims_spikes/images/{}.png".format(file_name_save))
    plt.close()
    #plt.show()
    #plt.savefig(r"D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\Thesis\comparison_analysis\Report_23_01\sinkhorn_measures\images\{}.png".format(title))

def read_datasets_in_group(file_path):
    datasets = []  # Initialize a dictionary to store dataset names and their data

    try:
        with h5py.File(file_path, 'r') as file:
            group = file['all_tensors']  # Access the desired group

            # Iterate through the datasets in the group
            for dataset_name, dataset in group.items():
                data = dataset[:]  # Extract the data from the dataset into a NumPy array
                datasets.append(torch.tensor(data))

    except Exception as e:
        print(f"Error: {str(e)}")

    return datasets

def pickl_reader(name):
    with open(name, 'rb') as file:
        gt_pc = pickle.load(file)
    return gt_pc

def random_sample(tensor, n):
    indices = torch.randperm(tensor.size(1))[: n]
    return tensor[:, indices]

def visualize_data_with_SD(gt_file, data_file, cond_num):
    ground_truth_pc = gt_file
    #discrete_cp = r"D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\Thesis\comparison_analysis\Report_23_01\extended_conditions\conditional_prior\discrete"
    data_path = "/home/goku454c/experiments_train_gen_conditional_prior/results/airplane_gen_model/random_rotated_stars/data"
    num_conds = 4
    #for i in range(num_conds):
    i = 0
    continuous_cp_tensor = read_datasets_in_group(os.path.join(data_path, data_file))[-1].squeeze(0)
    #discrete_cp_tensor = read_datasets_in_group(os.path.join(discrete_cp,  "discrete_cond_value_{}_sample_num_0.h5".format(i)))[-1].squeeze(0)
    
    cont_cp_tensor_SD = random_sample(continuous_cp_tensor, 2000)
    #disc_cp_tensor_SD = random_sample(discrete_cp_tensor, 2000) 
    gt_ind = int(cond_num[0]*3)
    gt_tensor = gt_file[gt_ind]['cloud']

    SD_cont = sinkhorn_loss(gt_tensor.T, cont_cp_tensor_SD.T, 0.01, 2000, 100).item()
    #wandb.log({"Sinkhorn loss": SD_cont})
    #SD_disc = sinkhorn_loss(gt_tensor.T, disc_cp_tensor_SD.T, 0.01, 2000, 100).item()
    scatter_plots(gt_tensor.detach().numpy(), "Ground truth", 'blue', "ground_truth_{}".format(str(cond_num)))
    scatter_plots(cont_cp_tensor_SD.detach().numpy(), "Continuous with SD {} (CP)".format(round(SD_cont, 3)), 'green', data_file[:-3])
    #scatter_plot_first_two_dimensions(gt_tensor, "Ground truth", "blue", "ground_truth_{}".format(i))
    #scatter_plot_2d_combinations(gt_tensor.detach().numpy(), "Ground Truth", 'blue', gt_file)
    #scatter_plot_2d_combinations(cont_cp_tensor_SD.detach().numpy(), "Continuous with SD {} (CP)".format(round(SD_cont, 3)), 'green', data_file)
    #scatter_plot_2d_combinations(disc_cp_tensor_SD.detach().numpy(), 'Discrete with SD {} (CP)'.format(round(SD_disc, 3)), 'green', "discrete_cond_{}_cp".format(i))

#gen_data = r"D:\Masters\Univerities\TU Dresden\Post_Admit\Studies\Thesis\comparison_analysis\Report_23_01\dimensions_study\Conditional_prior\Generated_data"
#visualize_data_with_SD(gen_data)

def visualize_data_with_SD_random(data_file, cond_num):
    ground_truth_pc = "/bigdata/hplsim/aipp/Gopal/data_npy/electron_data_processed.pkl"
    data_path = "/home/damoda95/Upload/results/airplane_gen_model/"
 
    continuous_cp_tensor = read_datasets_in_group(os.path.join(data_path, data_file))[-1].squeeze(0)
    #cont_cp_tensor_SD = random_sample(continuous_cp_tensor, 1000)
    cont_cp_tensor_SD = continuous_cp_tensor
    gt_list = pickl_reader(ground_truth_pc)
    gt_tensor = gt_list[int(8 + cond_num[2])]['cloud']

    SD_cont = sinkhorn_loss(gt_tensor.T, cont_cp_tensor_SD.T, 0.01, 5000, 100).item()
    #wandb.log({"Sinkhorn loss": SD_cont})

    min_val = [np.min(gt_tensor[_].detach().numpy()) for _ in range(6)]
    max_val = [np.max(gt_tensor[_].detach().numpy()) for _ in range(6)]
    
    # Set the same axis limits for both plots
    axis_limits = (min_val, max_val)

    #overlay_scatter_plots(gt_tensor.detach().numpy(), cont_cp_tensor_SD.detach().numpy(), "Continuous with SD {} (CP) with cond.{}, {}".format(round(SD_cont, 3), cond_num[0], cond_num[1]), 'blue', 'green', data_file[:-3])
    scatter_plots(gt_tensor.detach().numpy(), "Ground truth", 'blue', "ground_truth_{} cond {}".format(str(cond_num), cond_num), axis_limits)
    scatter_plots(cont_cp_tensor_SD.detach().numpy(), "Continuous with SD {} (CP) cond {}".format(round(SD_cont, 3), cond_num), 'green', "generated_data_{}".format(str(cond_num)), axis_limits)