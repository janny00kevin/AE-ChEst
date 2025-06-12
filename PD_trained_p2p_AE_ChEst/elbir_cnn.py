import torch
import torch.nn as nn

class cnn(nn.Module):
    def __init__(self, n_R=4, n_T=8, T=8, filt_size=3, num_channel=8):
        super(cnn, self).__init__()
        # Convolution 1 , input_shape=(:,3,4,8)
        self.cnn1 = nn.Conv2d(3,num_channel,filt_size) #output_shape=(:,256,2,6)
        # print("filt_size:",filt_size)
        # self.cnn2 = nn.Conv2d(256,256,3) #output_shape=(:,256,8,8)
        # self.cnn3 = nn.Conv2d(256,256,3) #output_shape=(:,256,6,6)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(num_channel*(n_R-filt_size+1)*(T-filt_size+1), 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2*n_R*n_T+2)
        # self.dp = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
                # Initialize conv layers and make them non-trainable
        # self._initialize_weights_and_freeze()
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        # out = self.cnn2(out)
        # out = self.cnn3(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.tanh(self.fc1(out))
        # out = self.dp(out)
        # out = self.tanh(self.fc2(out))
        # out = self.dp(out)
        out = self.fc3(out)
        return out
    
class cnn_InF_3fc(nn.Module):
    def __init__(self, n_R=4, n_T=8, T=8, filt_size=3, num_channel=8):
        super(cnn_InF_3fc, self).__init__()
        # Convolution 1 , input_shape=(:,3,4,8)
        self.cnn1 = nn.Conv2d(3,num_channel,filt_size) #output_shape=(:,256,2,6)
        # print("filt_size:",filt_size)
        # self.cnn2 = nn.Conv2d(256,256,3) #output_shape=(:,256,8,8)
        # self.cnn3 = nn.Conv2d(256,256,3) #output_shape=(:,256,6,6)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(num_channel*(n_R-filt_size+1)*(T-filt_size+1), 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2*n_R*n_T+2)
        # self.dp = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
                # Initialize conv layers and make them non-trainable
        # self._initialize_weights_and_freeze()
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        # out = self.cnn2(out)
        # out = self.cnn3(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.tanh(self.fc1(out))
        # out = self.dp(out)
        out = self.tanh(self.fc2(out))
        # out = self.dp(out)
        out = self.fc3(out)
        return out
    
class cnn2ch(nn.Module):
    def __init__(self, n_R=4, n_T=8, T=8, filt_size=3, num_channel=8):
        super(cnn2ch, self).__init__()
        # Convolution 1 , input_shape=(:,2,4,8)
        self.cnn1 = nn.Conv2d(2,num_channel,filt_size, padding='same') #output_shape=(:,256,2,6)
        # print("filt_size:",filt_size)
        # self.cnn2 = nn.Conv2d(256,256,3) #output_shape=(:,256,8,8)
        # self.cnn3 = nn.Conv2d(256,256,3) #output_shape=(:,256,6,6)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(num_channel*(n_R)*(T), 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2*n_R*n_T+2)
        # self.dp = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
                # Initialize conv layers and make them non-trainable
        # self._initialize_weights_and_freeze()
    
    def forward(self, x):
        # Convolution 1
        out = self.tanh(self.cnn1(x))
        # out = self.cnn2(out)
        # out = self.cnn3(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.tanh(self.fc1(out))
        # out = self.dp(out)
        # out = self.fc2(out)
        # out = self.dp(out)
        out = self.fc3(out)
        return out
    
class cnn2ch_2cv(nn.Module):
    def __init__(self, n_R=4, n_T=8, T=8, filt_size=3, num_channel=8):
        super(cnn2ch_2cv, self).__init__()
        # Convolution 1 , input_shape=(:,2,4,8)
        self.cnn1 = nn.Conv2d(2,num_channel,filt_size, padding='same') #output_shape=(:,256,2,6)
        # print("filt_size:",filt_size)
        self.cnn2 = nn.Conv2d(num_channel,num_channel,filt_size, padding='same')
        # self.cnn2 = nn.Conv2d(256,256,3) #output_shape=(:,256,8,8)
        # self.cnn3 = nn.Conv2d(num_channel,num_channel,filt_size, padding='same')
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(num_channel*(n_R)*(T), 1024)
        # self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2*n_R*n_T+2)
        # self.dp = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
                # Initialize conv layers and make them non-trainable
        # self._initialize_weights_and_freeze()
    
    def forward(self, x):
        # Convolution 1
        out = self.tanh(self.cnn1(x))
        out = self.tanh(self.cnn2(out))
        # out = self.tanh(self.cnn3(out))
        # out = self.cnn2(out)
        # out = self.cnn3(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.tanh(self.fc1(out))
        # out = self.dp(out)
        # out = self.fc2(out)
        # out = self.dp(out)
        out = self.fc3(out)
        return out
    
class cnn2ch_InF_3fc(nn.Module):
    def __init__(self, n_R=4, n_T=8, T=8, filt_size=3, num_channel=8):
        super(cnn2ch_InF_3fc, self).__init__()
        # Convolution 1 , input_shape=(:,2,4,8)
        self.cnn1 = nn.Conv2d(2,num_channel,filt_size) #output_shape=(:,256,2,6)
        # print("filt_size:",filt_size)
        # self.cnn2 = nn.Conv2d(256,256,3) #output_shape=(:,256,8,8)
        # self.cnn3 = nn.Conv2d(256,256,3) #output_shape=(:,256,6,6)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(num_channel*(n_R-filt_size+1)*(T-filt_size+1), 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2*n_R*n_T+2)
        # self.dp = nn.Dropout(p=0.5)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(p=0.5)
                # Initialize conv layers and make them non-trainable
        # self._initialize_weights_and_freeze()
    
    def forward(self, x):
        # Convolution 1
        out = self.tanh(self.cnn1(x))
        # out = self.cnn2(out)
        # out = self.cnn3(out)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.tanh(self.fc1(out))
        # out = self.dp(out)
        out = self.tanh(self.fc2(out))
        # out = self.dp(out)
        out = self.fc3(out)
        return out
    
class elbir(nn.Module):
    def __init__(self, n_R=4, n_T=8, T=8, filt_size=3, num_channel=256, input_channels=3, ReLU=False):
        super(elbir, self).__init__()

        self.is_relu = ReLU
        
        # Convolutional layers
        self.cnn1 = nn.Conv2d(input_channels, num_channel, kernel_size=filt_size, padding='same')
        # self.bn1 = nn.BatchNorm2d(num_filters)
        self.relu = nn.ReLU()
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.cnn2 = nn.Conv2d(num_channel, num_channel, kernel_size=filt_size, padding='same')
        # self.bn2 = nn.BatchNorm2d(num_filters)
        # self.relu2 = nn.ReLU()
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.cnn3 = nn.Conv2d(num_channel, num_channel, kernel_size=filt_size, padding='same')
        # self.bn3 = nn.BatchNorm2d(num_filters)
        # self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(num_channel*n_R*T, 1024)  # Adjust based on flattened dimension
        self.dp = nn.Dropout(p=0.5)
        
        # Fully connected layer 2
        self.fc2 = nn.Linear(1024, 2048)
        # self.dp2 = nn.Dropout(p=0.5)

        # Final output layer
        self.fc3 = nn.Linear(2048, 2*n_R*n_T+2)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu(out) if self.is_relu else out
        # Convolution 2
        out = self.cnn2(out)
        out = self.relu(out) if self.is_relu else out
        # Convolution 3
        out = self.cnn3(out)
        out = self.relu(out) if self.is_relu else out

        # Flatten for fully connected layers
        out = out.view(out.size(0), -1)
        
        # Fully connected layer 1
        out = self.fc1(out)
        out = self.relu(out) if self.is_relu else out
        out = self.dp(out)
        # Fully connected layer 2
        out = self.fc2(out)
        out = self.relu(out) if self.is_relu else out
        out = self.dp(out)

        # Output layer
        out = self.fc3(out)
        return out
