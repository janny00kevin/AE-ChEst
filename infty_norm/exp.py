from utils.p2p_ct_channels import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
groundChannel, receivedSignal, U_tilda, h_mean, h_std = importData(1000, 4, 36, 36, device, phase='train', channel='inf')
print(groundChannel.shape)
print(receivedSignal.shape)
print(U_tilda.shape)
print(h_mean.shape)
print(h_std.shape)