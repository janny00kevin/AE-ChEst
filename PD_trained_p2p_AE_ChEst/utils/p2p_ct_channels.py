import torch
import scipy.io as scio
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_path, '..')))
from utils.complex_utils import turn_real
# from utils.batch_khatri_rao import batch_khatri_rao
# from utils.get_IRS_coef import get_IRS_coef
# import time
import mat73


def import_data(data_size, n_R, n_T, T, SNR_lin, device, phase = 'train', channel='default'):

    W_Mean = 0
    sqrt2 = 2**0.5
    current_path = os.path.dirname(os.path.abspath(__file__))
    if not torch.is_tensor(SNR_lin):
        raise TypeError("SNR_lin should be a torch tensor")

    ## generate communication data to train the parameterized policy
    if channel.lower() == 'uma':
        if phase in ['train', 'val']:
            dataset_file_name = 'A_Training_Dataset_Eigen_UMa28_TisnT.mat'
        elif phase == 'test':
            dataset_file_name = 'A_Testing_Dataset_Eigen_UMa28_TisnT_3000.mat'
    elif channel.lower() == 'inf':
        if phase in ['train', 'val']:
            dataset_file_name = 'A_Training_Dataset_Eigen_InF25_TisnT.mat'
        elif phase == 'test':
            dataset_file_name = 'A_Testing_Dataset_Eigen_InF25_TisnT_3000.mat'
    else:
        raise NameError(f"{channel} is not a valid channel name")

    dataset_folder = os.path.abspath(os.path.join(current_path, '..', '..', 'p2p_ct_dataset'))
    dataset_file_path = os.path.join(dataset_folder, dataset_file_name)
    dataset = mat73.loadmat(dataset_file_path)
    h_ = torch.tensor(dataset['GroundChan']).to(device)

    # 90k data samples for training, 2k for validation
    if phase == 'train':
        h = torch.complex(h_[:data_size,:144],h_[:data_size,144:]).to(torch.complex64)
    elif phase == 'val':
        h = torch.complex(h_[h_.size(0)-data_size:,:144],h_[h_.size(0)-data_size:,144:]).to(torch.complex64)
    elif phase == 'test':
        h = torch.complex(h_[:,:144],h_[:,144:]).to(torch.complex64)
        # test_size = 3000
        # h = h[:test_size]
        # data_size = data_size*len(SNR_lin)  
        h = h.repeat(len(SNR_lin), 1, 1).reshape(data_size, n_R*n_T) # Repeat the channel for each SNR group (8)

    h_mean = h.mean()
    h_std = h.std()

    if phase == 'train':
        h = (h - h_mean) / h_std
    sgnl = h  # Transmitted signal (vectorized)

    Ps = (sgnl.abs()**2).mean() # Power of the transmitted signal
    Pn = Ps / SNR_lin # Noise power
    Pn = Pn.repeat_interleave(data_size*n_R*T*2//len(SNR_lin)).reshape(data_size, n_R*T, 2) # Repeat the noise power for each data sample groups (8)
    w = torch.view_as_complex(torch.normal(W_Mean, torch.sqrt(Pn))/sqrt2).to(device) # Generate noise
    y = sgnl + w # Received signal
    return turn_real(h), turn_real(y), h_mean, h_std # Return the channel, received signal, mean and std of the channel


# if __name__ == '__main__':
#     data_size = 2e3#9e4
#     n_R = 4
#     n_T = 36
#     T = 36
#     device = torch.device('cuda:1')  # 或 'cuda:0'
#     SNR_lin = torch.tensor([4, 2, 0, 2, 4, 6, 8, 10]).to(device)  # SNR in linear scale
#     phase = 'val'
#     channel = 'uma'

#     h, y, h_mean, h_std = importData(int(data_size), n_R, n_T, T, SNR_lin, device, phase, channel)
#     print("Channel shape:", h.shape)