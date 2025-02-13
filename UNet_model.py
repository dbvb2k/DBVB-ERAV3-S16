import torch
import torch.nn as nn

# Define the dice loss function
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

# Define the double convolution layer
class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

# Define the UNet model
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, use_maxpool=True, use_transpose=True):
        super().__init__()
        self.use_maxpool = use_maxpool
        self.use_transpose = use_transpose
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        self.bottleneck = DoubleConv(512, 1024)
        
        # Decoder path - fix the channel dimensions
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) if use_transpose else \
                  nn.Sequential(
                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                      nn.Conv2d(1024, 512, kernel_size=1)
                  )
        self.dec4 = DoubleConv(1024, 512)  # 512 + 512 input channels
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) if use_transpose else \
                  nn.Sequential(
                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                      nn.Conv2d(512, 256, kernel_size=1)
                  )
        self.dec3 = DoubleConv(512, 256)  # 256 + 256 input channels
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) if use_transpose else \
                  nn.Sequential(
                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                      nn.Conv2d(256, 128, kernel_size=1)
                  )
        self.dec2 = DoubleConv(256, 128)  # 128 + 128 input channels
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) if use_transpose else \
                  nn.Sequential(
                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                      nn.Conv2d(128, 64, kernel_size=1)
                  )
        self.dec1 = DoubleConv(128, 64)    # 64 + 64 input channels
        
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Downsampling
        if use_maxpool:
            self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            # Replace lambda with proper strided convolution
            self.down = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
            )
            self.current_down_idx = 0
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        
        # Reset down index at the start of forward pass if using strided conv
        if not self.use_maxpool:
            self.current_down_idx = 0
            
        # Modified downsampling for strided conv case
        if self.use_maxpool:
            e2 = self.enc2(self.down(e1))
            e3 = self.enc3(self.down(e2))
            e4 = self.enc4(self.down(e3))
            b = self.bottleneck(self.down(e4))
        else:
            e2 = self.enc2(self.down[0](e1))
            e3 = self.enc3(self.down[1](e2))
            e4 = self.enc4(self.down[2](e3))
            b = self.bottleneck(self.down[3](e4))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        
        return self.out_conv(d1) 