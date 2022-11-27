import torch
import torch.nn as nn

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    '''
    inconv only changes the number of channels
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            double_conv(in_ch, out_ch),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()
        self.bilinear=bilinear
        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch, 1),)
        else:
            self.up =  nn.ConvTranspose2d(in_channels=in_ch, out_channels=in_ch , kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1):
        x1 = self.up(x1)
        # x = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
        

class SimpleUnet(nn.Module):  # 1raw1of
    '''
    rawRange: Int, the idx of raw inputs to be predicted
    '''
    def __init__(self, features_root=128, raw_channel_num=128,  tot_raw_num=1, tot_of_num=1, border_mode='predict', rawRange=None, padding=True):
        super(SimpleUnet, self).__init__()

        assert tot_of_num <= tot_raw_num
        if border_mode == 'predict':
            self.raw_center_idx = tot_raw_num - 1
            self.of_center_idx = tot_of_num - 1
        else:
            self.raw_center_idx = (tot_raw_num - 1) // 2
            self.of_center_idx = (tot_of_num - 1) // 2
        if rawRange is None:
            self.rawRange = range(tot_raw_num)
        else:
            if rawRange < 0:
                rawRange += tot_raw_num
            assert rawRange < tot_raw_num
            self.rawRange = range(rawRange, rawRange+1)
        self.raw_channel_num = raw_channel_num
        self.tot_of_num = tot_of_num
        self.tot_raw_num = tot_raw_num
        self.raw_of_offset = self.raw_center_idx - self.of_center_idx
        
        self.padding = padding
        assert self.raw_of_offset >= 0

        if self.padding:
            in_channels = raw_channel_num
        else:
            in_channels = self.raw_channel_num * (tot_raw_num - 1)

        raw_out_channels = self.raw_channel_num

        self.inc = inconv(in_channels, features_root)
        self.down1 = down(features_root, features_root * 2)
        self.down2 = down(features_root * 2, features_root * 2)
        # self.down3 = down(features_root * 2, features_root * 2)
        # 0
        # self.up1 = up(features_root * 2, features_root * 2)
        self.up2 = up(features_root * 2, features_root * 2)
        self.up3 = up(features_root * 2, features_root)
        self.outc = outconv(features_root, raw_out_channels)

    def forward(self, x):
        # use incomplete inputs to yield complete inputs
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        
        # raw = self.up1(x4)
        raw = self.up2(x3)
        raw = self.up3(raw)
        raw_output = self.outc(raw)
        
        return raw_output
