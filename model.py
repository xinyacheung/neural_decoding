import torch
import torch.nn as nn
    
class spike_decode(nn.Module):
    def __init__(self, neuron_dim: int = 140, batch_dim: int = 45):
        super(spike_decode, self).__init__()
        self.linear1 = nn.Linear(neuron_dim, 512)
        self.bn = nn.BatchNorm1d(batch_dim)
        self.linear2 = nn.Linear(512,1024*3)
        self.to_image = nn.Unflatten(2, (3,32,32))
        
    def forward(self, x):
        #  [bs, time, neurons] -> [bs, time, embedding_size]
        x = self.linear1(x)
        x = self.bn(x)       
        x = self.linear2(x)
        x = self.to_image(x)
        
        return x.permute(0,2,1,3,4) #[bs, C, D, H, W]

class unet_decode(nn.Module):
    def __init__(self, channels: int = 3, hiddensize: int = 64, dropout: float=0.1):
        super(unet_decode, self).__init__()
        self.unet_encoder = nn.Sequential(
            nn.Conv3d(channels, hiddensize, kernel_size=(7,7,7), padding='same'),
            nn.BatchNorm3d(hiddensize),
            nn.ReLU(),
            nn.MaxPool3d([3,2,2]),
            nn.Conv3d(hiddensize, hiddensize*2, kernel_size=(5,5,5), padding='same'),
            nn.BatchNorm3d(hiddensize*2),
            nn.ReLU(),
            nn.MaxPool3d([3,2,2]),
            nn.Conv3d(hiddensize*2, hiddensize*4, kernel_size=(3,3,3), padding='same'),
            nn.BatchNorm3d(hiddensize*4),
            nn.ReLU(),
            nn.MaxPool3d([5,2,2]),
            nn.Conv3d(hiddensize*4, hiddensize*4, kernel_size=(3,3,3), padding='same'),
            nn.BatchNorm3d(hiddensize*4),
            nn.ReLU(),
            nn.MaxPool3d([1,2,2])
        )
        
        self.unet_decoder = nn.Sequential(
            nn.Upsample(scale_factor=tuple([1,2,2]), mode='nearest'),
            nn.Conv3d(hiddensize*4, hiddensize*4, kernel_size=(3,3,3), padding='same'),
            nn.BatchNorm3d(hiddensize*4),
            nn.ReLU(),
            nn.Upsample(scale_factor=tuple([5,2,2]), mode='nearest'),
            nn.Conv3d(hiddensize*4, hiddensize*2, kernel_size=(3,3,3), padding='same'),
            nn.BatchNorm3d(hiddensize*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=tuple([3,2,2]), mode='nearest'),
            nn.Conv3d(hiddensize*2, hiddensize, kernel_size=(5,5,5), padding='same'),
            nn.BatchNorm3d(hiddensize),
            nn.ReLU(),
            nn.Upsample(scale_factor=tuple([3,2,2]), mode='nearest'),
            nn.Conv3d(hiddensize, channels, kernel_size=(7,7,7), padding='same'),
            nn.BatchNorm3d(channels)
        )
    
    def forward(self, x):
        x = self.unet_encoder(x)
        x = self.unet_decoder(x)
        return x
        
class Decoder(nn.Module):
    def __init__(self, spikedecoder:spike_decode, unet:unet_decode):
        super(Decoder, self).__init__()
        self.spike_decoder = spikedecoder
        self.video_decoder = unet
        self.tanh = nn.Tanh()
        
    def decode(self, x):
        # [bs, neurons, time] -> [bs, neurons, embedding_dims] -> [bs, C3, D45, H32, W32]
        x = self.spike_decoder(x)
        x = self.video_decoder(x)
        x = (self.tanh(x)+1)/2 # Tanh scaled
        return x
    
def build_decoder(neuron_dim: int = 140) -> Decoder:
    spikedecoder = spike_decode(neuron_dim, batch_dim=45)
    unet = unet_decode(channels=3, hiddensize=48)
    video_decoder = Decoder(spikedecoder, unet)
    return video_decoder