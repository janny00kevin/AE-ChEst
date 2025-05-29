import torch
import torch.nn as nn
# from torch.nn import init
# import torch.nn.functional as F
# from torch import randn
from torch.linalg import norm
# import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import mat73
# from torch.autograd import Function
# import torch.nn.utils.prune as prune
import time

############ MAIN CONSOLE ############
mode = 'train'      # train or test
# mode = 'test'
dataset = 'InF'
version = '1L'
sub_version = 'cont_z_u'
note = version + '_' + dataset +'_' + sub_version
if dataset == 'InF':
    load_training_file = './data/A_Training_Dataset_Eigen_InF25_TisnT.mat'
    load_testing_file = './data/A_Testing_Dataset_Eigen_InF25_TisnT_3000.mat'
    scaling = 1e4
    lambda_sparsity = 1e-8  # Sparsity penalty weight
else:
    load_training_file = './data/A_Training_Dataset_Eigen_UMa28_TisnT.mat'
    load_testing_file = './data/A_Testing_Dataset_Eigen_UMa28_TisnT_3000.mat'
    scaling = 1e5
    lambda_sparsity = 1e-6  # Sparsity penalty weight

model_name = 'model/model_parameters_AE_ChEst_' + note + '.pth'
fig_name_01 = "fig/training_curve_" + note + ".png"
fig_name_02 = "fig/ADMM_iteration_" + note + ".png"
pre_trained_model = 'model/model_init_AEChEst_' + version + '_' + dataset + '.pth'
pre_trained = 0
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# n_layer = 10
epsilon = 1e3
rho = 100
k_max = 5
j_max = 10
data_size = 10000
batch_size = 8000
epochs = 3000
# scaling_ISTA = 1
t = [0.001]
lamb = [10,20,30]
test_network = 1
test_conven = 0

######################################

def print_grad(network):
    for name, param in network.named_parameters():
        if param.grad is not None:
            print(f"Gradient - {name}: {param.grad.device}")
        else:
            print(f"Gradient - {name}: None")

def print_n_param(network):
    for name, param in network.named_parameters():
        print(f"{name}: {param.size()} -> {param.numel()}")
    print('')
    print('\ntotal learnable params: ',sum(p.numel() for p in network.parameters() if p.requires_grad))

def plot_01(totalr, totals, totalrho, fig_name):
    plt.clf()
    plt.plot(totals, label='dual residual', color='red')
    plt.plot(totalr, label='primal residual', color='blue')
    plt.plot(totalrho, label='rho', color='green')
    plt.plot([1]*len(totalr), 'k--', label='convergence threshold')
    plt.xlabel('ADMM iteration, k')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(fig_name)

def plot_training_validation_loss(train_loss, test_loss, fig_name):
    plt.clf()
    plt.plot(train_loss, label='training loss', color='blue')
    plt.plot(test_loss, label='validation loss', color='red')
    plt.xlabel('epoch')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(fig_name)

def plot_one(x, fig_name):
    plt.clf()
    plt.plot(x, color='blue')
    plt.xlabel('iteration')
    plt.yscale('log')
    plt.grid()
    plt.savefig(fig_name)

def LMMSE_solver(y, A, x_groundtruth, C_noise, MC):
    C_x = cov_mat(x_groundtruth, MC)
    Ax = A @ x_groundtruth
    C_Ax = cov_mat(Ax, MC)
    C_y = C_Ax + C_noise
    C_xy = C_x @ torch.conj(A.T)
    temp = C_xy @ torch.linalg.pinv(C_y)
    mean_x = torch.reshape(torch.mean(x_groundtruth, dim=1), (-1 ,1))
    mean_y = A @ mean_x
    x_lmmse = torch.zeros_like(x_groundtruth)
    for m in range(MC):
        y_ = torch.reshape(y[:,m], (-1,1)).to(torch.complex64)
        x_lmmse[:,m] = torch.squeeze(mean_x + temp @ (y_ - mean_y))
    return x_lmmse

def LS_solver(y, A):
    temp = torch.linalg.pinv(torch.conj(A.T) @ A) @ torch.conj(A.T)
    return temp @ y.to(torch.complex64)

def cov_mat(A, MC):
    mean_A = torch.reshape(torch.mean(A, dim=1), (-1, 1))
    mean_A_sqr = mean_A @ torch.conj(mean_A.T)
    C_A = 0
    for m in range(MC):
        m1 = torch.reshape(A[:,m], (-1, 1)) 
        m2 = torch.reshape(torch.conj(A[:,m]), (1, -1)) 
        C_A += (m1 @ m2 - mean_A_sqr)
    return C_A / MC

def gen_noisy_sgn(A, SNR):
    len, MC = A.shape
    A_pwr = torch.zeros(len, 1)
    for k in range(len):
        a = A[k,:].reshape(-1,1)
        A_pwr[k] = torch.real(a.H @ a / MC)
    noisepwr = A_pwr * 10 ** (-SNR/10)
    C_noise = torch.diag(torch.squeeze(noisepwr)).to(device)
    C_noise_stdev = torch.linalg.cholesky(C_noise).to(device)
    random = torch.complex(torch.randn(A.shape), torch.randn(A.shape)).to(device)
    noise =  (1/2**(1/2)) * C_noise_stdev.to(torch.complex128) @ random.to(torch.complex128)
    Y = A + noise
    return Y, C_noise

def update_theta(network, receivedSignal, x_old, u, rho, j_max=1000, thres=1e-7):
    totalloss = [999]
    counter = 0
    for j in range(j_max):
        h_hat, alpha = network(receivedSignal)
        loss_all,s,f,s_ratio = loss_theta_function(alpha, h_hat, x_old, u, rho)
        totalloss.append(loss_all.item())
        res = totalloss[-1] - totalloss[-2]
        counter = counter + 1 if res > 0 else 0
        if (np.abs(res) < thres or counter > 5) and (j > 100): break
        else:
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()
    plot_one(totalloss[1:], 'fig/scratch15.png')
    return j

def loss_theta_function(latent_sparse, output, x, u, rho):
    z_sparse_ = to_matrix(latent_sparse)
    output_ = to_matrix(output)
    x_ = to_matrix(x)
    u_ = to_matrix(u)

    x_detached = x_.detach()
    u_detached = u_.detach()

    sparsity_term = torch.sum(torch.abs(z_sparse_))
    fidelity_term = torch.norm(x_detached - X_tilda @ output_ + u_detached, 2)**2
    loss_all = sparsity_term + rho/2 * fidelity_term
    sparsity_ratio = sparsity_term / loss_all * 100
    return loss_all, sparsity_term.item(), fidelity_term.item(), sparsity_ratio.item()

def l1_loss(model, lambda_l1=0.001):
    """ Computes L1 norm of weights and adds to the loss """
    l1_reg = 0.0
    for param in model.parameters():
        l1_reg += torch.sum(torch.abs(param))  # Sum of absolute values of weights
    return lambda_l1 * l1_reg

def to_matrix(a):
    len_a = a.shape[1] // 2
    return torch.complex(a[:,:len_a].T, a[:,len_a:].T)

def to_data_form(a_):
    return torch.concat((torch.real(a_), torch.imag(a_)), dim=0).T

def soft_thres(x, lamb):
    len_x = x.shape[1] // 2
    x_ = torch.complex(x[:,:len_x].T, x[:,len_x:].T)
    abs_x = torch.abs(x_)
    x_sparse_ = x_ * torch.relu(abs_x - lamb) / abs_x
    x_sparse_.real[torch.isnan(x_sparse_.real)] = 0
    x_sparse_.imag[torch.isnan(x_sparse_.imag)] = 0
    x_sparse = torch.concat((x_sparse_.real, x_sparse_.imag), dim=0).T
    return x_sparse

def get_residual_norm(h_hat, z_kk, temp):
    M = h_hat.shape[1]

    r = z_kk - h_hat.detach()
    norm_r = [torch.norm(r[:,ind], 2).item() for ind in range(M)]
    
    norm_s = []
    for ind in range(M):
        grad = torch.autograd.grad(temp[ind], network.parameters(), grad_outputs=torch.ones_like(temp[ind]), retain_graph=True)
        norm_s_sq = 0
        for j in range(len(grad)):
            norm_s_sq += torch.norm(grad[j])**2
        norm_s.append(torch.sqrt(norm_s_sq).item())

    return norm_r, norm_s

def update_rho(norm_r, norm_s, rho, mu=10, tau_inc=2, tau_dec=2):
    # ratio = norm_r / norm_s
    if max(norm_r) > mu*max(norm_s): return rho*tau_inc
    elif max(norm_s) > mu*max(norm_r): return rho/tau_dec
    else: return rho

def convergence_criterion(norm_r, norm_s, z, h_hat, temp, n_param, ep_abs=1e-3, ep_rel=1e-3):
    p,b = z.shape

    ep_prim = []
    for ind in range(b):
        ep_prim.append((np.sqrt(p)*ep_abs + ep_rel*max(torch.norm(z[:,ind], 2), torch.norm(h_hat[:,ind], 2))).item())

    norm_grad = []
    for ind in range(b):
        grad = torch.autograd.grad(temp[ind], network.parameters(), grad_outputs=torch.ones_like(temp[ind]), retain_graph=True)
        norm_grad_sq = 0
        for j in range(len(grad)):
            norm_grad_sq += torch.norm(grad[j]) ** 2
        norm_grad.append(torch.sqrt(norm_grad_sq))
    ep_dual = [(np.sqrt(n_param)*ep_abs + ep_rel*q).item() for q in norm_grad]

    primal_convergence = [x / y for x, y in zip(norm_r, ep_prim)] 
    dual_convergence = [x / y for x, y in zip(norm_s, ep_dual)] 

    sum_primal_converge = sum(x > 1 for x in primal_convergence)
    sum_dual_converge = sum(x > 1 for x in dual_convergence)

    all_primal_converge = all(x <= 1 for x in primal_convergence)
    all_dual_converge = all(x <= 1 for x in dual_convergence)

    # return primal_convergence, dual_convergence
    return all_primal_converge and all_dual_converge, max(primal_convergence), max(dual_convergence), sum_primal_converge, sum_dual_converge

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder01 = nn.Linear(input_dim, input_dim)
        # self.encoder02 = nn.Linear(input_dim, input_dim)
        self.decoder01 = nn.Linear(input_dim, input_dim)
        # self.decoder02 = nn.Linear(input_dim, input_dim)
        # self.relu = nn.ReLU()  # Use ReLU or sigmoid based on your needs
        self.lamb = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, x):
        # Encoder
        # latent = self.encoder02(soft_thres(self.encoder01(x), torch.abs(self.lamb)))
        latent = self.encoder01(x)
        latent_sparse = soft_thres(latent, torch.abs(self.lamb))
        # latent_sparse = self.relu(latent)
        # Decoder
        # reconstructed = self.decoder02(soft_thres(self.decoder01(latent_sparse), torch.abs(self.lamb)))
        reconstructed = self.decoder01(latent_sparse)
        return reconstructed, latent_sparse
    
# class SparseAutoencoder(nn.Module):
#     def __init__(self, input_dim, n_layer):
#         super(SparseAutoencoder, self).__init__()

#         encoder_layers = []
#         for _ in range(n_layer-1):
#             encoder_layers.append(nn.Linear(input_dim, input_dim))
#             # encoder_layers.append(nn.BatchNorm1d(input_dim))
#             # encoder_layers.append(nn.ReLU()) 
#         encoder_layers.append(nn.Linear(input_dim, input_dim))
#         # encoder_layers.append(nn.BatchNorm1d(input_dim))
#         self.encoder = nn.Sequential(*encoder_layers)

#         self.lamb = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))

#         decoder_layers = []
#         for _ in range(n_layer-1):
#             decoder_layers.append(nn.Linear(input_dim, input_dim))
#             # decoder_layers.append(nn.BatchNorm1d(input_dim))
#             # decoder_layers.append(nn.ReLU())
#         decoder_layers.append(nn.Linear(input_dim, input_dim))  # Output layer
#         self.decoder = nn.Sequential(*decoder_layers)

#     def forward(self, x):
#         latent = self.encoder(x)
#         latent_sparse = soft_thres(latent, self.lamb)
#         # latent_sparse = latent
#         reconstructed = self.decoder(latent_sparse)
#         return reconstructed, latent_sparse
    
# class SparseAutoencoder(nn.Module):
#     def __init__(self, input_dim):
#         super(SparseAutoencoder, self).__init__()
#         self.hidden_dim = 1024
#         self.enfc1 = nn.Linear(input_dim, self.hidden_dim)
#         self.enfc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.enfco = nn.Linear(self.hidden_dim, input_dim)
#         self.defc1 = nn.Linear(input_dim, self.hidden_dim)
#         self.defc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.defco = nn.Linear(self.hidden_dim, input_dim)
#         self.tanh = nn.Tanh()
#         # self.encoder01 = nn.Linear(input_dim, input_dim)
#         # self.decoder01 = nn.Linear(input_dim, input_dim)
#         # self.relu = nn.ReLU()  # Use ReLU or sigmoid based on your needs
#         self.lamb = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

#     def forward(self, x):
#         # Encoder
#         # latent = self.encoder02(soft_thres(self.encoder01(x), torch.abs(self.lamb)))
#         latent = self.enfco(self.tanh(self.enfc2(self.tanh(self.enfc1(x)))))
#         latent_sparse = soft_thres(latent, torch.abs(self.lamb))
#         # latent_sparse = self.relu(latent)
#         # Decoder
#         # reconstructed = self.decoder02(soft_thres(self.decoder01(latent_sparse), torch.abs(self.lamb)))
#         reconstructed = self.defco(self.tanh(self.defc2(self.tanh(self.defc1(latent_sparse)))))
#         return reconstructed, latent_sparse

def solve_for_x(y, m, epsilon):
    """
    Fast closed-form projection for complex matrices onto a constraint ball.
    solve argmin_x  ||x - y||_2^2  subject to ||x - m||_2^2 <= epsilon
    """
    diff = y - m
    norm_sq = torch.sum(torch.abs(diff) ** 2)

    if norm_sq <= epsilon:
        return y  # Already inside the ball

    return m + torch.sqrt(epsilon / norm_sq) * diff

torch.manual_seed(2) 
torch.set_printoptions(threshold=10000)

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        # Implement data loading logic here
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])

nTx = 36 # number of transmitters
nRx = 4 # number of receivers

note = str(nRx) + 'x' + str(nTx)

# sent signal (pilot)
factor = 1
X = torch.eye(nTx, dtype=torch.cfloat)
X = X.repeat(1, factor)
X_tilda = torch.kron(X.T.contiguous(), torch.eye(nRx)).to(device)

if mode == 'train':
    Training_Dataset = mat73.loadmat(load_training_file)
    U_tilda = torch.tensor(Training_Dataset['U_tilda'])
    rand_indices = torch.randperm(100000)
    selected_indices = rand_indices[:data_size]
    sparseChannel = Training_Dataset['SparseChan'][selected_indices,:]
    receivedSignal = Training_Dataset['ReceivedSig'][selected_indices,:]
    groundChannel = Training_Dataset['GroundChan'][selected_indices,:]
    n_samples = sparseChannel.shape[0]
    scaling = 1/torch.from_numpy(groundChannel).std()
    # print("groundChannel shape: ", groundChannel.shape, "groundChannel type: ", groundChannel.dtype)
    print("scaling: ", scaling)

    # split the data into training and test sets
    receivedSignal_Train, receivedSignal_Test, groundChannel_Train, groundChannel_Test = train_test_split(receivedSignal, groundChannel, test_size=0.2, random_state=0)

    dataset_train = MyDataset((receivedSignal_Train, groundChannel_Train))
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataset_test = MyDataset((receivedSignal_Test, groundChannel_Test))
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    len_h = groundChannel.shape[1] // 2
    # network = SparseAutoencoder(len_h*2, n_layer).to(device)
    network = SparseAutoencoder(len_h*2).to(device)
    min_lr = 5e-7
    lossListTrain = []
    lossListTest = []
    mask = torch.eye(nRx*nTx).to(device)
    best_loss = float('inf')
    n_param = sum(p.numel() for p in network.parameters() if p.requires_grad)

    ## PRE-TRAINED MODEL FOR FEASIBLE INITIALIZATION ##
    if pre_trained == 1:
        print("\n=== Starting pre-training model for feasible initialization ===")
        init_model_fig_name = 'fig/init_model_' + version + '_' + dataset +'.png'
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)      # lr=1e-5 (slow)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)   # patience=2, factor=0.1
        
        lossListTrain = []
        lossListTest = []
        for e in range(epochs):
            network.train()
            avgTrainLoss = 0 
            for batch in tqdm(dataloader_train):
                receivedSignal,_ = batch
                receivedSignal = receivedSignal.to(torch.float32).to(device) * scaling
                
                h_hat, alpha = network(receivedSignal)
                loss_discrepancy = torch.mean(torch.pow(h_hat - receivedSignal, 2))
                alpha_ = torch.complex(alpha[:,:144], alpha[:,144:])
                sparsity_penalty = lambda_sparsity * torch.sum(torch.abs(alpha_))
                loss_all = loss_discrepancy + sparsity_penalty

                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()
                avgTrainLoss += loss_all.item()
            avgTrainLoss = avgTrainLoss * batch_size / len(dataset_train)
            scheduler.step(avgTrainLoss)
            lossListTrain.append(avgTrainLoss)

            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < min_lr:
                print(f"Stopping training as learning rate reached {current_lr:.2e}")
                break

            avgTestLoss = 0
            network.eval()
            with torch.no_grad():
                for batch in dataloader_test:

                    receivedSignal, _ = batch
                    receivedSignal = receivedSignal.to(torch.float32).to(device) * scaling

                    h_hat, alpha = network(receivedSignal)
                    loss_discrepancy = torch.mean(torch.pow(h_hat - receivedSignal, 2))
                    alpha_ = torch.complex(alpha[:,:144], alpha[:,144:])
                    sparsity_penalty = lambda_sparsity * torch.sum(torch.abs(alpha_))
                    loss_all = loss_discrepancy + sparsity_penalty

                    avgTestLoss += loss_all.item()

            avgTestLoss = avgTestLoss*(batch_size)/len(dataset_test)
            if(best_loss>avgTestLoss):
                torch.save(network.state_dict(), pre_trained_model) 
                best_loss = avgTestLoss
            lossListTest.append(avgTestLoss)
            print("  Epoch {:d}\t| Training loss: {:.4e}  | Testing loss: {:.4e} --- ratio: {:.3f}".format(e+1, avgTrainLoss, avgTestLoss, avgTrainLoss/avgTestLoss))
            plot_training_validation_loss(lossListTrain, lossListTest, init_model_fig_name)
        print("\n=== Finished model initialization ===\n\n")
    else:
        network.load_state_dict(torch.load(pre_trained_model))
        network = network.to(device)    
        print("\n=== Model initialization successfully loaded ===")
    
    print("\n=== Starting ADMM training algorithm ===\n")
    ## START TRAINING WITH ADMM ALGORITHM ##
    plt.ion()
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1, verbose=True)
    totalr = []
    totals = []
    totalrho = []
    u = torch.zeros((batch_size, len_h*2), dtype=torch.float).to(device)
    x_new = torch.zeros((1, 288), dtype=torch.float32).to(device)
    for e in range(epochs):
        avgTrainLoss = 0 
        network.train()
        for batch in tqdm(dataloader_train):

            receivedSignal, groundChannel = batch
            receivedSignal = receivedSignal.to(torch.float32).to(device) * scaling
            groundChannel = groundChannel.to(torch.float32).to(device) * scaling
            batch_size = receivedSignal.shape[0]
            e_ = []
            
            # x_new, _ = network(receivedSignal)
            # u = torch.zeros((batch_size, len_h*2), dtype=torch.float).to(device)
            k = 0
            converge = False
            while k < k_max:
                k += 1
                x_old = x_new
                x_old_ = to_matrix(x_old)

                # start_time = time.time()
                j_used = update_theta(network, receivedSignal, x_old, u, rho, j_max)
                # end_time = time.time()
                # print(f"Time taken for update_theta: {end_time - start_time:.4f} seconds")

                # start_time = time.time()
                h_hat, alpha = network(receivedSignal)
                alpha_ = to_matrix(alpha)              
                h_hat_ = to_matrix(h_hat)
                u_ = to_matrix(u)
                receivedSignal_ = to_matrix(receivedSignal)

                x_new_ = solve_for_x(X_tilda @ h_hat_ - u_, receivedSignal_, epsilon).detach()
                x_new = to_data_form(x_new_)
                # end_time = time.time()
                # print(f"Time taken for solve_for_x: {end_time - start_time:.4f} seconds")
                

                # start_time = time.time()
                u_ = (u_ + x_new_ - X_tilda @ h_hat_).detach()
                u = to_data_form(u_)
                
                # temp = torch.trace(rho * (x_old_ - x_new_).T @ h_hat_).real
                # temp = rho * torch.sum((x_old_ - x_new_) * h_hat_).real
                temp = sum((rho * (x_old_ - x_new_) * h_hat_).real)
                norm_r, norm_s = get_residual_norm(h_hat_, x_new_, temp)
                del temp
                torch.cuda.empty_cache()
                # end_time = time.time()
                # print(f"Time taken for get_residual_norm: {end_time - start_time:.4f} seconds")

                # start_time = time.time()
                # temp = torch.trace((rho*u_).T @ h_hat_).real
                # temp = rho * torch.sum(u_ * h_hat_).real
                temp = sum((rho * u_ * h_hat_).real)
                converge, normalized_prim, normalized_dual, sum_primal_converge, sum_dual_converge = convergence_criterion(norm_r, norm_s, x_new_, h_hat_, temp, n_param, ep_abs=1e-2, ep_rel=1e-2)
                # end_time = time.time()
                # print(f"Time taken for convergence_criterion: {end_time - start_time:.4f} seconds")

                # start_time = time.time()
                del temp
                torch.cuda.empty_cache()

                totalr.append(normalized_prim)
                totals.append(normalized_dual)
                totalrho.append(rho)

                if converge: 
                    plot_01(totalr, totals, totalrho, fig_name_02)
                    break
                rho = update_rho(norm_r, norm_s, rho, mu=1e2, tau_inc=1.1, tau_dec=1.2)

                # if k % 10 == 0: 
                plot_01(totalr, totals, totalrho, fig_name_02)  # plot every 10 iterations
                end_time = time.time()
                # print(f"Time taken for plotting: {end_time - start_time:.4f} seconds")
            loss_e = nn.MSELoss()(h_hat, groundChannel)
            avgTrainLoss += loss_e.item()
        
        avgTrainLoss = avgTrainLoss * batch_size / len(dataset_train)
        scheduler.step(avgTrainLoss)
        lossListTrain.append(avgTrainLoss)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < min_lr:
            print(f"Stopping training as learning rate reached {current_lr:.2e}")
            break

        avgTestLoss = 0
        network.eval()
        with torch.no_grad():
            for batch in dataloader_test:

                receivedSignal, groundChannel = batch
                receivedSignal = receivedSignal.to(torch.float32).to(device) * scaling
                groundChannel = groundChannel.to(torch.float32).to(device) * scaling
                batch_size = receivedSignal.shape[0]

                h_hat, alpha = network(receivedSignal)
                # z_sparse_ = to_matrix(alpha)
                # output_ = to_matrix(h_hat)
                # receivedSignal_ = to_matrix(receivedSignal)

                # sparsity_term = torch.sum(torch.abs(z_sparse_))
                # fidelity_term = torch.norm(receivedSignal_ - X_tilda @ output_, 2)**2

                # loss_all,sparse_loss,fidelity_loss, sparsity_ratio = loss_theta_function(alpha, h_hat, x_old, u, rho)
                # loss_e = torch.norm(receivedSignal_ - X_tilda @ h_hat_, 2)
                loss_e = nn.MSELoss()(h_hat, groundChannel)
                # avgTestLoss += loss_all.item()
                avgTestLoss += loss_e.item()

        avgTestLoss = avgTestLoss * batch_size / len(dataset_test)
        lossListTest.append(avgTestLoss)

        if best_loss > avgTestLoss:
            torch.save(network.state_dict(), model_name) 
            print("Model saved")
            best_loss = avgTestLoss
        
        print("  Epoch {:d}\t| Training loss: {:.4e}  | Testing loss: {:.4e} --- ratio: {:.3f}".format(e+1, avgTrainLoss, avgTestLoss, avgTrainLoss/avgTestLoss))
        plot_training_validation_loss(lossListTrain, lossListTest, fig_name_01)

else:
    print('test')
    try:
        Testing_Dataset = mat73.loadmat(load_testing_file)
        groundChannel = torch.tensor(Testing_Dataset['GroundChan'])
        U_tilda = torch.tensor(Testing_Dataset['U_tilda']).to(device)
        n_sample, len_h = int(groundChannel.shape[0]), int(groundChannel.shape[1]/2)
    except:
        Testing_Dataset = mat73.loadmat(load_testing_file)
        groundChannel_ = torch.tensor(Testing_Dataset['h_all'])
        groundChannel = to_data_form(groundChannel_)
    # else:
    #     with h5py.File(load_testing_file, "r") as f:
    #         groundChannel = f["GroundChan"][:]
    #         U_tilda = f["A_tilda"][:]

    #     groundChannel = torch.tensor(groundChannel)
    #     U_tilda = torch.tensor(U_tilda).to(device)
    #     n_sample, len_h = int(groundChannel.shape[0]), int(groundChannel.shape[1]/2)

    SNR = [i for i in range(-4, 11, 2)]
    # SNR = [-10, 0, 15]

    # groundChannel = torch.complex(groundChannel[:,:len_h].T, groundChannel[:,len_h:].T).to(device)
    MC, len_h_2 = groundChannel.shape
    MC = 3000
    len_h = int(len_h_2/2)
    network_1st = SparseAutoencoder(len_h_2).to(device)
    network_1st.load_state_dict(torch.load(model_name, map_location=device))
    network_1st = network_1st.to(device)
    network_1st.eval()

    groundChannel_ = to_matrix(groundChannel)
    print("groundChannel shape: ", groundChannel.shape, "groundChannel type: ", groundChannel_.dtype)
    print("groundChannel_ shape: ", groundChannel_.shape, "groundChannel_ type: ", groundChannel_.dtype)
    scaling = 1/groundChannel.std()
    print("scaling: ", scaling)

    h_all_ = groundChannel_[:,:MC].to(torch.complex64).to(device)

    ISTA_NMSE_all = torch.zeros(len(SNR), MC, len(t), len(lamb)).to(device)
    count_ISTA = torch.zeros(len(SNR), len(t), len(lamb)).to(device)
    n_nonzero = torch.zeros(len(SNR), len(t), len(lamb)).to(device)
    LMMSE_NMSE_all = torch.zeros(len(SNR), MC).to(device)
    LS_NMSE_all = torch.zeros(len(SNR), MC).to(device)
    LISTA_NMSE_all_1st = torch.zeros(len(SNR), MC).to(device)

    for snr_ind in range(len(SNR)):

        print("==== SNR: {} dB ====".format(SNR[snr_ind]))

        Xh = X_tilda.to(device) @ h_all_
        y_, C_noise =  gen_noisy_sgn(Xh, SNR[snr_ind])

        h_lmmse = LMMSE_solver(y_, X_tilda, h_all_, C_noise, MC)
        h_ls = LS_solver(y_, X_tilda)
        
        # LMMSE AND LS
        for m in range(MC):
            LMMSE_NMSE_all[snr_ind, m] = ((norm(h_lmmse[:,m] - h_all_[:,m], 2)/norm(h_all_[:,m], 2))**2).item()
            LS_NMSE_all[snr_ind, m] = ((norm(h_ls[:,m] - h_all_[:,m], 2)/norm(h_all_[:,m], 2))**2).item()

        # if test_conven:
        #     # ISTA-LS
        #     if 1:
        #         XU = X_tilda.to(torch.complex128) @ U_tilda 
        #         for t_ind in range(len(t)):
        #             for lamb_ind in range(len(lamb)):
        #                 hc_all_k, count, nonzero = ISTA_solver(y, XU, t[t_ind], lamb[lamb_ind], scaling_ISTA, device)
        #                 h_all_k = U_tilda @ hc_all_k
        #                 error_ista = h_all - h_all_k
        #                 for m in range(MC):
        #                     ISTA_NMSE_all[snr_ind, m, t_ind, lamb_ind] = (torch.norm(error_ista[:,m])/torch.norm(h_all[:,m]))**2
        #                 count_ISTA[snr_ind,t_ind,lamb_ind] = count
        #                 n_nonzero[snr_ind,t_ind,lamb_ind] = nonzero                

        if test_network:
            with torch.no_grad(): 
                # FIRST MODEL TESTING
                y = to_data_form(y_).to(torch.float32) * scaling
                h_hat, alpha = network_1st(y)
                h_hat_ = to_matrix(h_hat) / scaling
                error_lista = h_all_ - h_hat_
                for m in range(MC):
                    LISTA_NMSE_all_1st[snr_ind, m] = ((torch.norm(error_lista[:,m])/torch.norm(h_all_[:,m]))**2).item()

        LMMSE_NMSE_k = torch.mean(LMMSE_NMSE_all[snr_ind, :])
        LS_NMSE_k = torch.mean(LS_NMSE_all[snr_ind, :])
        if test_conven:
            ISTA_NMSE_k = torch.mean(ISTA_NMSE_all, dim=1)[snr_ind]
        if test_network:
            LISTA_NMSE_1st_k = torch.mean(LISTA_NMSE_all_1st[snr_ind, :])
        print("\n--- SUMMARY ---")
        print("=== SNR: {} === ".format(SNR[snr_ind]))
        print("\tLS ---------> NMSE:  {:.4f} dB ".format(10*torch.log10(LS_NMSE_k).item()))
        print("\tLMMSE ------> NMSE:  {:.4f} dB ".format(10*torch.log10(LMMSE_NMSE_k).item()))
        if test_conven:
            temp_min = torch.min(10*torch.log10(ISTA_NMSE_k))
            temp_argmin = torch.argmin(10*torch.log10(ISTA_NMSE_k))
            temp_i, temp_j= (temp_argmin)//len(lamb), (temp_argmin)%len(lamb)
            print("\tISTA -------> NMSE*: {:.4f} dB \tCount: {:.2f}".format(temp_min, count_ISTA[snr_ind, temp_i, temp_j]))
            print("\t\twith t: {} and lambda: {}".format(t[temp_i], lamb[temp_j]))
        if test_network:
            print("\tLISTA_1st ---> NMSE: {:.4f} dB".format(10*torch.log10(LISTA_NMSE_1st_k).item()))
        
    LS_NMSE =torch.squeeze(torch.mean(LS_NMSE_all, dim=1)) 
    LMMSE_NMSE =torch.squeeze(torch.mean(LMMSE_NMSE_all, dim=1))
    if test_conven:
        ISTA_NMSE = torch.squeeze(torch.mean(ISTA_NMSE_all, dim=1))
    if test_network:
        LISTA_1st_NMSE = torch.squeeze(torch.mean(LISTA_NMSE_all_1st, dim=1))

    print("\n\n=== OVERALL SUMMARY ===")
    print("SNR = {};".format(list(SNR)))
    print("LS_NMSE = {};".format([round(e, 4) for e in (10*torch.log10(LS_NMSE)).tolist()]))
    print("LMMSE_NMSE = {};".format([round(e, 4) for e in (10*torch.log10(LMMSE_NMSE)).tolist()]))

    if test_conven:
        for k in range(len(lamb)):
            print("ISTA_NMSE_{} = {};".format(lamb[k], [round(e, 4) for e in (10*torch.log10(ISTA_NMSE[:,k])).tolist()]))
        print(10*torch.log10(ISTA_NMSE))
    if test_network:
        print("LISTA_NMSE_1st = {};".format([round(e, 4) for e in (10*torch.log10(LISTA_1st_NMSE)).tolist()]))

    print('\n====== END ======\n')