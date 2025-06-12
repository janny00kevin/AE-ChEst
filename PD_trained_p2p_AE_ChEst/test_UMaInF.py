import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
# from utils.batch_kronecker import batch_kronecker
from utils.complex_utils import turn_real, turn_cplx, vec
from utils.p2p_ct_channels import import_data
# from utils.get_IRS_coef import get_IRS_coef
from utils.LMMSE import LMMSE_solver
import os
import pandas as pd

torch.manual_seed(0)

device = torch.device("cuda:0")
# device1 = torch.device("cuda:2")
script_dir = os.path.dirname(os.path.abspath(__file__))
n_T = 36
# n_I = 8
n_R = 4
T = 36
channel = 'uma'
print('channel:', channel)

if channel == 'inf':
    # MLPs
    # filename1 = os.path.join(script_dir, 'trained_model', '4.646_eps1e+01_inf_AE_lr1e-03_[288, 1024, 288]_ep68.pt')
    # checkpoint1 = torch.load(filename1, weights_only=False, map_location=device)
    filename2 = os.path.join(script_dir, 'trained_model', '8.435_eps15.0_inf_AE_lr1e-03_[288, 1024, 288]_ep128.pt')
    checkpoint2 = torch.load(filename2, weights_only=False, map_location=device)
    # filename3 = os.path.join(script_dir, 'trained_model/comaring_models', '0.748_SP_InF_nmlz_lr1e-04_[288, 1024, 1024, 290]_ep2500.pt')
    # checkpoint3 = torch.load(filename3, weights_only=False, map_location=device)

    # filename4 = os.path.join(script_dir, 'trained_model', '1.282_8SNR_inf_5MLP_psi_h_lr5e-05_[256, 1024, 258]_ep129.pt')
    # checkpoint4 = torch.load(filename4, weights_only=False, map_location=device)

    # filename5 = os.path.join(script_dir, 'trained_model', '1.237_8SNR_inf_10MLP_psi_h_lr1e-04_[256, 1024, 258]_ep75.pt')
    # checkpoint5 = torch.load(filename5, weights_only=False, map_location=device)

    # # channelNets
    # filename6 = os.path.join(script_dir, 'trained_model/comaring_models', '6.644_SP_elwRe_InF_lr1e-04_[288, 3, 256, 1024, 290]_ep3.pt')
    # checkpoint6 = torch.load(filename6, weights_only=False, map_location=device)

    # # ISTA-Nets
    # filename9 = os.path.join(script_dir, 'trained_model/comaring_models', '0.907_inf_ISTA_psi_h_lr1e-03_ep29.pt')
    # checkpoint9 = torch.load(filename9, weights_only=False, map_location=device)
elif channel == 'uma':
    # MLPs
    # filename1 = os.path.join(script_dir, 'trained_model', '4.646_eps1e+01_inf_AE_lr1e-03_[288, 1024, 288]_ep68.pt')
    # checkpoint1 = torch.load(filename1, weights_only=False, map_location=device)
    filename2 = os.path.join(script_dir, 'trained_model', '0.007_eps0.4_uma_AE_lr1e-03_[288, 1024, 288]_ep154.pt')
    checkpoint2 = torch.load(filename2, weights_only=False, map_location=device)
    # filename3 = os.path.join(script_dir, 'trained_model/comaring_models', '0.748_SP_InF_nmlz_lr1e-04_[288, 1024, 1024, 290]_ep2500.pt')
    # checkpoint3 = torch.load(filename3, weights_only=False, map_location=device)



# logits_net1 = checkpoint1['logits_net'].to(device)
logits_net2 = checkpoint2['logits_net'].to(device)
# logits_net3 = checkpoint3['logits_net'].to(device) 
# logits_net4 = checkpoint4['logits_net'].to(device) 
# logits_net5 = checkpoint5['logits_net'].to(device)  
# tnn = checkpoint10['tnn'].to(device)

# logits_net4 = checkpoint4['logits_net'].to(device)
# logits_net5 = checkpoint5['logits_net'].to(device)
# logits_net6 = checkpoint6['logits_net'].to(device) 

# logits_net7 = checkpoint7['logits_net'].to('cuda:0')
# logits_net8 = checkpoint8['logits_net'].to('cuda:1') 
# logits_net9 = checkpoint9['logits_net'].to('cuda:2')

sqrt2 = 2**.5
SNR_dB = torch.arange(-4,10.1,2)
SNR_lin = 10**(SNR_dB/10.0).to(device)
# NMSE_1 = torch.zeros_like(SNR_lin) # MLP 1
NMSE_2 = torch.zeros_like(SNR_lin) # MLP 2
# NMSE_3 = torch.zeros_like(SNR_lin)
# NMSE_4 = torch.zeros_like(SNR_lin) # channelNet
# NMSE_5 = torch.zeros_like(SNR_lin) 
# NMSE_6 = torch.zeros_like(SNR_lin)
# NMSE_7 = torch.zeros_like(SNR_lin) # ISTA-Net 
# NMSE_8 = torch.zeros_like(SNR_lin) 
if channel == 'inf':
    # ISTA-LS-Net-conv
    NMSE_1 = torch.tensor([7.2354, 4.5106, 3.3117, 3.0683, 2.8981, 2.6805, 2.7235, 2.6907])
    # nonparametric PD MLP
    NMSE_3 = torch.tensor([10.026526, 6.3835974, 4.138638, 2.2707534, 1.3489263, 0.010249095, -1.0680854, -2.0834825])
    # channelNet
    NMSE_6 = torch.tensor([12.8535385, 10.731388, 8.933279, 6.544038, 4.9803357, 3.2596698, 2.0081658, 1.0957559])
    # ISTA-Net
    NMSE_9 = torch.tensor([10.174, 7.5188, 5.9056, 4.7055, 3.3911, 2.3565, 0.9998, -0.0795])
elif channel == 'uma':
    NMSE_1 = torch.tensor([-16.4166, -18.578, -20.839, -22.5987, -24.5692, -26.3854, -28.0583, -30.4035])
    NMSE_3 = torch.tensor([-15.111146, -17.936468, -20.302063, -22.017296, -23.901176, -26.337791, -27.933994, -29.81211])
    NMSE_6 = torch.tensor([-15.2742405, -18.023226, -20.412523, -21.978827, -23.768524, -25.9894, -27.409706, -29.203741])
    NMSE_9 = torch.tensor([-12.7118, -15.6712, -18.2572, -20.8751, -22.8632, -25.0052, -26.6167, -28.6719])
# NMSE_10 = torch.zeros_like(SNR_lin)

# NMSE_LS_i = torch.zeros_like(SNR_lin) # LS
# NMSE_LM_i = torch.zeros_like(SNR_lin) # sampled LMMSE
# NMSE_LS_d = torch.zeros_like(SNR_lin) # LS
# NMSE_LM_d = torch.zeros_like(SNR_lin) # sampled LMMSE
NMSE_LS = torch.zeros_like(SNR_lin) # LS
NMSE_LM = torch.zeros_like(SNR_lin) # sampled LMMSE

# test_size = int(3e3)#2.4e4
test_size = int(2.4e4)#3e3
datasize_per_SNR = test_size//len(SNR_lin)
# test_data_size = datasize_per_SNR*len(SNR_dB)

### import testing data ###
h_test, y_test, h_mean, h_std = import_data(test_size, n_R, n_T, T, SNR_lin, device, phase = 'test', channel=channel)
# print(h_mean, h_std)
# _, y_test_d, _, _ = import_data(test_size, n_R, n_T, T, SNR_lin, device, phase = 'test', channel=channel)
# _, y_test_h, _, _ = import_data(test_size, n_R, n_T, T, SNR_lin, device, IRScoef='h', phase = 'test', channel=channel)
h_test = h_test.reshape(len(SNR_lin), datasize_per_SNR, n_R*n_T*2)
y_test = y_test.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
# y_test_h = y_test_h.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
# y_test_d = y_test_d.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)
# Y_test_nmlz_i = (torch.view_as_complex(y_test_i.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std
# Y_test_nmlz_d = (torch.view_as_complex(y_test_d.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std
Y_test_nmlz = (torch.view_as_complex(y_test.reshape(len(SNR_lin), datasize_per_SNR, n_T*n_R, T//n_T, 2)) - h_mean)/h_std
# _, y_test_TNN, _, _ = importData(test_size, n_R, n_I, n_T, T, SNR_lin, device, IRScoef=Psi, phase = 'test', channel=channel)
# y_test_TNN = y_test_TNN.reshape(len(SNR_lin), datasize_per_SNR, n_R*T*2)

# y_test = add_noise(tnn(turnCplx(h_test)), SNR_lin, test_size, n_R, T, device)
# logits_test = logits_net(turnReal(y_test - h_mean)/h_std)


def LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, snr, h_test, y_test):
    """
    Computes the Least Squares (LS) and Linear Minimum Mean Square Error (LMMSE) norms for given test data.
    Args:
        device (torch.device): The device to perform computations on (e.g., 'cpu' or 'cuda').
        datasize_per_SNR (int): The size of the data per Signal-to-Noise Ratio (SNR).
        n_R (int): Number of receive antennas.
        n_T (int): Number of transmit antennas.
        T (int): Number of time slots.
        n_I (int): Number of IRS elements.
        snr (float): Signal-to-Noise Ratio.
        IRS_coef_type (str): Type of IRS coefficients.
        h_test (torch.Tensor): The test channel matrix.
        y_test (torch.Tensor): The received signal matrix.
    Returns:
        tuple: A tuple containing:
            - norm_LS (torch.Tensor): The norm of the LS estimation error.
            - norm_LM (torch.Tensor): The norm of the LMMSE estimation error.
    """
    # Psi = get_IRS_coef(IRS_coef_type,n_R,n_I,n_T,T).to(device)
    # tbPsi = batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64)
    ### LS
    LS_solution = turn_cplx(y_test)
    # Compute the norm of the difference
    norm_LS = torch.norm((turn_cplx(h_test) - LS_solution), dim=1)**2

    ### LMMSE
    # tbPsi = batch_kronecker(Psi.T, torch.eye(n_T*n_R).to(device)).to(torch.complex64)
    X_tild = torch.complex(torch.eye(n_R*n_T), torch.zeros(n_R*n_T, n_R*n_T)).to(device)
    Sgnl = turn_cplx(h_test)
    Ps = (Sgnl.abs()**2).mean(dim=0)
    Pn = Ps / snr
    cov_n = torch.diag(Pn)
    lmmse = LMMSE_solver(turn_cplx(y_test).T, X_tild, turn_cplx(h_test).T, cov_n, datasize_per_SNR).T
    norm_LM = torch.norm((turn_cplx(h_test) - lmmse), dim=1)**2
    return norm_LS,norm_LM

D = torch.cat((torch.eye(n_R*n_T), torch.zeros(n_R*n_T,1)),1).to(torch.complex64).to(device)

pbar = tqdm(total = len(SNR_dB))
for idx, snr in enumerate(SNR_lin):
    snr = snr.unsqueeze(0).to(device)
    # IRS_coef_type = checkpoint1['IRS_coe_type']

    with torch.no_grad():
        # # Y_nmlz = (y.reshape(data_size,n_R,T)-h_mean2)/h_std2
        # logits1 = turn_cplx(logits_net1(turn_real(turn_cplx(y_test[idx])-h_mean)/h_std)[0])*h_std + h_mean
        # norm_1 = torch.norm(turn_cplx(h_test[idx]) - logits1, dim=1)**2

        logits2 = turn_cplx(logits_net2(turn_real(turn_cplx(y_test[idx])-h_mean)/h_std)[0])*h_std + h_mean
        norm_2 = torch.norm(turn_cplx(h_test[idx]) - logits2, dim=1)**2

        # logits3 = logits_net3(turn_real(turn_cplx(y_test[idx])-h_mean)/h_std)
        # test_tbh_cplx = turn_cplx(logits3)*h_std + h_mean
        # norm_3 = torch.norm(turn_cplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # logits4 = logits_net4(turn_real(turn_cplx(y_test_h[idx])-h_mean)/h_std)
        # test_tbh_cplx = turn_cplx(logits4)*h_std + h_mean
        # norm_4 = torch.norm(turn_cplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # logits5 = logits_net5(turn_real(turn_cplx(y_test_h[idx])-h_mean)/h_std)
        # test_tbh_cplx = turn_cplx(logits5)*h_std + h_mean
        # norm_5 = torch.norm(turn_cplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # y_test_TNN = IRS_ct_channels_TNN.add_noise(tnn(turnCplx(h_test[idx])), snr, y_test_h[idx].shape[0], n_R, T, device)
        # logits10 = logits_net10(turnReal(y_test_TNN - h_mean)/h_std)
        # test_tbh_cplx = turnCplx(logits10)*h_std + h_mean
        # norm_10 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # y_test_TNN = IRS_ct_channels_TNN.add_noise(tnn(turnCplx(h_test[idx])), snr, y_test_h[idx].shape[0], n_R, T, device)
        # logits10 = logits_net10(turnReal(turnCplx(y_test_TNN[idx]) - h_mean)/h_std)
        # test_tbh_cplx = turnCplx(logits10)*h_std + h_mean
        # norm_10 = torch.norm(turnCplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T, dim=1)**2

        # ### channelNet
        # logits4 = logits_net4(torch.stack([Y_test_nmlz_i[idx].real,Y_test_nmlz_i[idx].imag,Y_test_nmlz_i[idx].abs()],dim=1))
        # test_tbh_cplx = turnCplx(logits4)*h_std + h_mean
        # norm_4 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits5 = logits_net5(torch.stack([Y_test_nmlz_d[idx].real,Y_test_nmlz_d[idx].imag,Y_test_nmlz_d[idx].abs()],dim=1))
        # test_tbh_cplx = turnCplx(logits5)*h_std + h_mean
        # norm_5 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits6 = logits_net6(torch.stack([Y_test_nmlz[idx].real,Y_test_nmlz[idx].imag,Y_test_nmlz[idx].abs()],dim=1))
        # test_tbh_cplx = torch.view_as_complex(logits6.reshape(datasize_per_SNR, n_T*n_R+1, 2))
        # norm_6 = torch.norm(turn_cplx(h_test[idx]) - D.matmul(test_tbh_cplx.T).T*h_std-h_mean, dim=1)**2
        # logits4 = logits_net4(torch.stack([Y_nmlz.real,Y_nmlz.imag,Y_nmlz.abs()],dim=1))
        # test_tbh_cplx = torch.view_as_complex(logits4.reshape(data_size, n_T*n_R+1, 2))
        # norm_4 = torch.norm(h - D.matmul(test_tbh_cplx.T).T*h_std4-1*h_mean4, dim=1)**2

        # ### ISTA-Net
        # # torch.cuda.empty_cache()
        # logits7, _ = logits_net7((turnReal(turnCplx(y_test_i[idx])-h_mean)/h_std))
        # test_tbh_cplx = turnCplx(logits7).to(device)*h_std + h_mean
        # norm_7 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits8, _ = logits_net8((turnReal(turnCplx(y_test_d[idx])-h_mean)/h_std))
        # test_tbh_cplx = turnCplx(logits8).to(device)*h_std + h_mean
        # norm_8 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2
        
        # logits9, _ = logits_net9((turnReal(turnCplx(y_test_h[idx])-h_mean)/h_std))
        # test_tbh_cplx = turnCplx(logits9).to(device)*h_std + h_mean
        # norm_9 = torch.norm(turnCplx(h_test[idx]) - test_tbh_cplx, dim=1)**2

        '''
        LS and LMMSE numerical solution for different IRS coefficient matrix
        '''

        # norm_LS_i, norm_LM_i = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_i[idx], IRS_coef_type='i')
        # norm_LS_d, norm_LM_d = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, n_I, snr, h_test[idx], y_test_d[idx], IRS_coef_type='d')
        norm_LS, norm_LM = LS_LMMSE(device, datasize_per_SNR, n_R, n_T, T, snr, h_test[idx], y_test[idx])

        # logits3 = logits_net3(torch.stack([Y_nmlz.real,Y_nmlz.imag,Y_nmlz.abs()],dim=1))
        # test_tbh_cplx = turnCplx(logits3)
        # norm_3 = torch.norm(h - D.matmul(test_tbh_cplx.T).T*h_std2-h_mean2, dim=1)**2
        # logits4 = logits_net4(torch.stack([Y_nmlz.real,Y_nmlz.imag,Y_nmlz.abs()],dim=1))
        # test_tbh_cplx = turnCplx(logits4)
        # norm_4 = torch.norm(h - D.matmul(test_tbh_cplx.T).T*h_std2-h_mean2, dim=1)**2


        # NMSE_1[idx] = 10*torch.log10((norm_1 / torch.norm(h_test[idx], dim=1)**2).mean())
        NMSE_2[idx] = 10*torch.log10((norm_2 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_3[idx] = 10*torch.log10((norm_3 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_4[idx] = 10*torch.log10((norm_4 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_5[idx] = 10*torch.log10((norm_5 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_4[idx] = 10*torch.log10((norm_4 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_5[idx] = 10*torch.log10((norm_5 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_6[idx] = 10*torch.log10((norm_6 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_7[idx] = 10*torch.log10((norm_7 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_8[idx] = 10*torch.log10((norm_8 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_9[idx] = 10*torch.log10((norm_9 / torch.norm(h_test[idx], dim=1)**2).mean())
        # NMSE_10[idx] = 10*torch.log10((norm_10 / torch.norm(h_test[idx], dim=1)**2).mean())

    # NMSE_LS_i[idx] = 10*torch.log10((norm_LS_i / torch.norm(h_test[idx], dim=1)**2).mean())
    # NMSE_LM_i[idx] = 10*torch.log10((norm_LM_i / torch.norm(h_test[idx], dim=1)**2).mean())
    # NMSE_LS_d[idx] = 10*torch.log10((norm_LS_d / torch.norm(h_test[idx], dim=1)**2).mean())
    # NMSE_LM_d[idx] = 10*torch.log10((norm_LM_d / torch.norm(h_test[idx], dim=1)**2).mean())
    NMSE_LS[idx] = 10*torch.log10((norm_LS / torch.norm(h_test[idx], dim=1)**2).mean())
    NMSE_LM[idx] = 10*torch.log10((norm_LM / torch.norm(h_test[idx], dim=1)**2).mean())
    
    # plt.text(10*torch.log10(snr)-1,NMSE_LM_i[idx]-1, f'({NMSE_LM_i[idx].item():.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr)-1,NMSE_2[idx]-1, f'({10**(NMSE_2[idx].item()/10):.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr),NMSE_10[idx], f'({10**(NMSE_10[idx].item()/10):.2f})')  ## plot linear NMSE value 
    # plt.text(10*torch.log10(snr),NMSE_10[idx], f'({(NMSE_10[idx].item()):.2f})')
    pbar.update(1)


# plt.plot(SNR_dB, NMSE_LS_i.to('cpu'), label='LS w/ I', linewidth=1, linestyle='--', marker='o', color="tab:blue")
# plt.plot(SNR_dB, NMSE_LS_d.to('cpu'), label='LS w/ D', linewidth=1, linestyle=':', marker='o', color="tab:blue")
plt.plot(SNR_dB, NMSE_LS.to('cpu'), label='LS', linewidth=1, linestyle='-', marker='o', color="tab:blue")
# plt.plot(SNR_dB, NMSE_LM_i.to('cpu'), label='LMMSE w/ I', linewidth=1, linestyle='--', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE_LM_d.to('cpu'), label='LMMSE w/ D', linewidth=1, linestyle=':', marker='o', color="tab:red")
plt.plot(SNR_dB, NMSE_LM.to('cpu'), label='LMMSE', linewidth=1, linestyle='-', marker='o', color="tab:red")
# plt.plot(SNR_dB, NMSE5,'-x', label='model-based PD LS', linewidth=1 ,color="tab:red")
# plt.plot(epochs, testing_loss_N.reshape(num_epochs,num_minibatch).mean(dim=1).to("cpu"), label='validation loss')

# plt.plot(SNR_dB, NMSE_4.to('cpu'), label='channelNet w/ I ', linewidth=1, linestyle='--', marker='x', color="tab:green")  ###
# plt.plot(SNR_dB, NMSE_5.to('cpu'), label='channelNet w/ D ', linewidth=1, linestyle=':', marker='x', color="tab:green")  ###
plt.plot(SNR_dB, NMSE_6.to('cpu'), label='channelNet', linewidth=1, linestyle='-', marker='o', color="tab:green")  ###

# plt.plot(SNR_dB, NMSE_7.to('cpu'), label='ISTANet w/ I ', linewidth=1, linestyle='--', marker='x', color="tab:brown")  ###
# plt.plot(SNR_dB, NMSE_8.to('cpu'), label='ISTANet w/ D ', linewidth=1, linestyle=':', marker='x', color="tab:brown")  ###
plt.plot(SNR_dB, NMSE_9.to('cpu'), label='ISTANet', linewidth=1, linestyle='-', marker='o', color="tab:brown")  ###

plt.plot(SNR_dB, NMSE_1.to('cpu'), label='ISTA-LS-Net-conv', linewidth=1, linestyle='-', marker='o', color="tab:pink")
plt.plot(SNR_dB, NMSE_3.to('cpu'), label='nonparametric PD MLP', linewidth=1, linestyle='-', marker='o', color="tab:orange")  ###
# plt.plot(SNR_dB, NMSE_1.to('cpu'), label='parametric PD AE $\\varepsilon=90$', linewidth=1, linestyle='-', marker='x', color="tab:gray")  ###
plt.plot(SNR_dB, NMSE_2.to('cpu'), label='parametric PD AE $\\varepsilon=0.4$', linewidth=1, linestyle='-', marker='x', color="black")  ###

# plt.plot(SNR_dB, NMSE_4.to('cpu'), label='5 layers MLP w/ H', linewidth=1, linestyle='-', marker='x', color="black")  ###

# plt.plot(SNR_dB, NMSE_5.to('cpu'), label='10 layers MLP w/ H', linewidth=1, linestyle=':', marker='x', color="black")  ###
# plt.plot(SNR_dB.flip(0), NMSE_10.to('cpu'), label='NP PD w/ TNN', linewidth=1, linestyle='-', marker='x', color="tab:brown")  ###

# plt.plot(SNR_dB, NMSE_3,'-x', label='channelNet w/o act. func.')  ###
# plt.plot(SNR_dB, NMSE_4,'-x', label='channelNet w/ act. func.')  ###

if channel == 'inf':
    plt.suptitle("PD trained MIMO nmlz ChEst vs SNR in InF2.5")
elif channel == 'uma':
    plt.suptitle("PD trained MIMO nmlz ChEst vs SNR in UMa28")
# plt.title(' $[n_R,n_T]$:[4,36], MLP size:[288, 2048, 2048, 576] ')

# plt.suptitle("MMSE based PD with PG channel estimator")
# plt.title('size:%s' %([2*n_R*T]+checkpoint1['hidden_sizes']+[2*n_R*n_T]))
plt.title('$[n_R,n_T,T]:[%s,%s,%s], datasize: %s$' %(n_R,n_T,T,test_size))
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE(dB)')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize='small')
plt.grid(True)

data = {
    "SNR_dB": SNR_dB.to('cpu'), "LS": NMSE_LS.to('cpu'), "LMMSE": NMSE_LM.to('cpu'),
    "AE 2": NMSE_2.to('cpu'), "MLP": NMSE_3.to('cpu'),
    "channelNet": NMSE_6.to('cpu'), "ISTANet": NMSE_9.to('cpu')
}
df = pd.DataFrame(data)
save_path = os.path.join(script_dir, 'ChEsts_testing_performance', '%s_AE_vs_CN_IN___.pdf'%(channel)) #  %(IRS_coef_type)
# plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig(save_path, bbox_inches = 'tight')   ###
df.to_csv(os.path.splitext(save_path)[0] + '.csv', index=False)

