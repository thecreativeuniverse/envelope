
import torch
import torch.nn as nn
import torch.nn.init as init
from koipond.util.fun import downsample
import numpy as np
import scipy.ndimage as ndimage
from scipy.signal import decimate
import scipy.interpolate as interp

#TODO
class KoiNet(nn.Module):
  
  def get_version():
       return "4.0"

  # define model elements
  def __init__(self, n_inputs):
        super().__init__()

        self.n_inputs = n_inputs

        kernel_size = 7

        # RAW VIEW
        self.raw_conv_1 = nn.Conv1d(1, 32, kernel_size=kernel_size)
        init.xavier_normal_(self.raw_conv_1.weight)
        init.zeros_(self.raw_conv_1.bias)
        self.raw_relu_1 = nn.LeakyReLU()
        self.dropout_1 = nn.Dropout(p=0.2)
        self.raw_conv_2 = nn.Conv1d(32, 16, kernel_size=kernel_size)
        init.xavier_normal_(self.raw_conv_2.weight) 
        init.zeros_(self.raw_conv_2.bias)
        self.raw_relu_2 = nn.LeakyReLU()
        self.dropout_2 = nn.Dropout(p=0.2)
        # TODO: margarita suggests kernel_size=n/p (n=n_inputs, p={2,3,5}, 
        # pooling factor. may be too big for what im doing?? idk)
        # (src: https://towardsdatascience.com/how-to-use-convolutional-neural-networks-for-time-series-classification-56b1b0a07a57)
        self.raw_pool_1 = nn.MaxPool1d(kernel_size=kernel_size, stride=2)
        self.raw_conv_3 = nn.Conv1d(16, 32, kernel_size=kernel_size)
        init.xavier_normal_(self.raw_conv_3.weight)
        init.zeros_(self.raw_conv_3.bias)
        self.raw_conv_4 = nn.Conv1d(32, 32, kernel_size=kernel_size)
        init.xavier_normal_(self.raw_conv_4.weight)
        init.zeros_(self.raw_conv_4.bias)
        self.raw_relu_2 = nn.LeakyReLU()
        self.dropout_3 = nn.Dropout(p=0.2)
        # TODO: same as above pooling 
        self.raw_pool_2 = nn.MaxPool1d(kernel_size=kernel_size, stride=2)

      #   # SMOOTHED VIEW
      #   self.n_smooth_views = 5
      #   self.smooth_conv_1 = nn.Conv1d(self.n_smooth_views, 32, kernel_size=kernel_size)
      #   init.xavier_normal_(self.smooth_conv_1.weight)
      #   init.zeros_(self.smooth_conv_1.bias)
      #   self.smooth_conv_2 = nn.Conv1d(32, 16, kernel_size=kernel_size)
      #   init.xavier_normal_(self.smooth_conv_2.weight)
      #   init.zeros_(self.smooth_conv_2.bias)
      #   self.smooth_relu_1 = nn.LeakyReLU()
      #   self.smooth_pool_1 = nn.MaxPool1d(kernel_size=kernel_size, stride=2)
      #   self.smooth_conv_3 = nn.Conv1d(16, 32, kernel_size=kernel_size)
      #   init.xavier_normal_(self.smooth_conv_3.weight)
      #   init.zeros_(self.smooth_conv_3.bias)
      #   self.smooth_conv_4 = nn.Conv1d(32, 32, kernel_size=kernel_size)
      #   init.xavier_normal_(self.smooth_conv_4.weight)
      #   init.zeros_(self.smooth_conv_4.bias)
      #   self.smooth_relu_2 = nn.LeakyReLU()
      #   self.smooth_pool_1 = nn.MaxPool1d(kernel_size=kernel_size, stride=2)
        
      #   # DOWN SAMPLED VIEW
      #   self.n_downsample_views = 5
      #   self.downsample_conv_1 = nn.Conv1d(self.n_downsample_views, 32, kernel_size=kernel_size)
      #   init.xavier_normal_(self.downsample_conv_1.weight)
      #   init.zeros_(self.downsample_conv_1.bias)
      #   self.downsample_conv_2 = nn.Conv1d(32, 16, kernel_size=kernel_size)
      #   init.xavier_normal_(self.downsample_conv_2.weight)
      #   init.zeros_(self.downsample_conv_2.bias)
      #   self.downsample_relu_1 = nn.LeakyReLU()
      #   self.downsample_pool_1 = nn.MaxPool1d(kernel_size=kernel_size, stride=2)
      #   self.downsample_conv_3 = nn.Conv1d(16, 32, kernel_size=kernel_size)
      #   init.xavier_normal_(self.downsample_conv_3.weight)
      #   init.zeros_(self.downsample_conv_3.bias)
      #   self.downsample_conv_4 = nn.Conv1d(32, 32, kernel_size=kernel_size)
      #   init.xavier_normal_(self.downsample_conv_4.weight)
      #   init.zeros_(self.downsample_conv_4.bias)
      #   self.downsample_relu_2 = nn.LeakyReLU()
      #   self.downsample_pool_2 = nn.MaxPool1d(kernel_size=kernel_size, stride=2)
        
      #   # CONCATENATION
      #   #nothing needed for init declaring

        # FULL CONVOLUTION
        self.full_convolution_1 = nn.Conv1d(32, 64, kernel_size=kernel_size)
        init.xavier_normal_(self.full_convolution_1.weight)
        init.zeros_(self.full_convolution_1.bias)
        self.full_relu1 = nn.LeakyReLU()
        self.dropout_4 = nn.Dropout(p=0.2)
        self.full_convolution_2 = nn.Conv1d(64, 64, kernel_size=kernel_size)
        init.xavier_normal_(self.full_convolution_2.weight)
        init.zeros_(self.full_convolution_2.bias)
        self.full_relu2 = nn.LeakyReLU()
        self.dropout_5 = nn.Dropout(p=0.2)
        self.full_pool_1 = nn.MaxPool1d(kernel_size=kernel_size, stride=2)

        # FULLY CONNECTED
        flattened_size = 7104
        self.fc_1 = nn.Linear(flattened_size,2048)
        init.normal_(self.fc_1.weight)
        init.normal_(self.fc_1.bias)
        self.fc_2 = nn.Linear(2048, 1024)
        init.normal_(self.fc_2.weight)
        init.normal_(self.fc_2.bias)
        self.fc_3 = nn.Linear(1024, 1)
        init.normal_(self.fc_3.weight)
        init.normal_(self.fc_3.bias)

        # ACTIVATION
        self.activation = nn.Sigmoid()


  # forward propagate input
  def forward(self, x):
        # RAW VIEW
      #   raw_x = x.clone()
        x = self.raw_conv_1(x)
        x = self.raw_relu_1(x)
        x = self.dropout_1(x)
        x = self.raw_conv_2(x)
        x = self.raw_relu_2(x)
        x = self.dropout_2(x)
        x = self.raw_pool_1(x)

        x = self.raw_conv_3(x)
        x = self.raw_conv_4(x)
        x = self.raw_relu_2(x)
        x = self.dropout_3(x)
        x = self.raw_pool_2(x)

      #   # SMOOTHED VIEW
      #   # remove irrelevant 1D channel size; will be self.n_smooth_views
      #   smoothed_x = x.clone().view(x.shape[0], x.shape[2])
      #   smoothed_x = torch.tensor(np.array([ndimage.gaussian_filter1d(smoothed_x, i) for i in np.linspace(1, 15, self.n_smooth_views)])).float()
      #   # reshape n channels to be second in shape
      #   smoothed_x = smoothed_x.permute(1,0,2)

      #   smoothed_x = self.smooth_conv_1(smoothed_x)
      #   smoothed_x = self.smooth_conv_2(smoothed_x)
      #   smoothed_x = self.smooth_relu_1(smoothed_x)
      #   smoothed_x = self.smooth_pool_1(smoothed_x)
      #   smoothed_x = self.smooth_conv_3(smoothed_x)
      #   smoothed_x = self.smooth_conv_4(smoothed_x)
      #   smoothed_x = self.smooth_relu_2(smoothed_x)
      #   smoothed_x = self.smooth_pool_1(smoothed_x)
        
      #   # DOWN SAMPLED VIEW
      #   # TODO this might need to be reshaped/interpolated to same dimensions..
      #   # remove irrelevant 1D channel size; will now be self.n_downsample_views
      #   downsampled_x = x.clone().view(x.shape[0], x.shape[2])
      #   downsampled_x = torch.tensor(np.array([downsample(x=np.arange(0, self.n_inputs, 1), ys=downsampled_x, q=round(i)) for i in np.linspace(1, 40, self.n_downsample_views)])).float()
      #   # reshape n channels to be second in shape
      #   downsampled_x = downsampled_x.permute(1,0,2)

      #   downsampled_x = self.downsample_conv_1(downsampled_x)
      #   downsampled_x = self.downsample_conv_2(downsampled_x)
      #   downsampled_x = self.downsample_relu_1(downsampled_x)
      #   downsampled_x = self.downsample_pool_1(downsampled_x)
      #   downsampled_x = self.downsample_conv_3(downsampled_x)
      #   downsampled_x = self.downsample_conv_4(downsampled_x)
      #   downsampled_x = self.downsample_relu_2(downsampled_x)
      #   downsampled_x = self.downsample_pool_2(downsampled_x)
        
      #   # CONCATENATION
      #   concatenated_x = torch.cat((raw_x, smoothed_x, downsampled_x), 2)

        # FULL CONVOLUTION
        x = self.full_convolution_1(x)
        x = self.full_relu1(x)
        x = self.dropout_4(x)
        x = self.full_convolution_2(x)
        x = self.full_relu2(x)
        x = self.dropout_5(x)
        x = self.full_pool_1(x)

        x = torch.flatten(x, 1)
        # FULLY CONNECTED
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.fc_3(x)

        # ACTIVATION
        output = self.activation(x)
        return output
