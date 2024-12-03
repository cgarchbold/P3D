import torch
import torch.nn as nn
import torch.nn.functional as F

# Networks
def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)

class CNNGenerator256(nn.Module):
    def __init__(self, z_dim=10, num_classes=10, label_embed_size=5, channels=3, conv_dim=64):
        super(CNNGenerator256, self).__init__()

        if num_classes != None:
            self.conditional = True
            self.tconv1 = conv_block(z_dim + 12, conv_dim * 32, pad=0, transpose=True)
        else:
            self.tconv1 = conv_block(z_dim, conv_dim * 32, pad=0, transpose=True)
            self.conditional = False

        self.tconv2 = conv_block(conv_dim * 32, conv_dim * 16, transpose=True)
        self.tconv3 = conv_block(conv_dim * 16, conv_dim * 8, transpose=True)
        self.tconv4 = conv_block(conv_dim * 8, conv_dim * 4, transpose=True)
        self.tconv5 = conv_block(conv_dim * 4, conv_dim * 2, transpose=True)
        self.tconv6 = conv_block(conv_dim * 2, conv_dim, transpose=True)
        self.tconv7 = conv_block(conv_dim, channels, transpose=True, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label=None):
        x = x.reshape([x.shape[0], -1, 1, 1])

        if self.conditional:
            label = label.reshape([label.shape[0], 12, 1,1])
            x = torch.cat((x, label), dim=1)

        x = F.relu(self.tconv1(x))
        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = F.relu(self.tconv4(x))
        x = F.relu(self.tconv5(x))
        x = F.relu(self.tconv6(x))
        x = torch.tanh(self.tconv7(x))
        return x


class CNNDiscriminator256(nn.Module):
    def __init__(self, num_classes=10, channels=3, conv_dim=64):
        super(CNNDiscriminator256, self).__init__()
        self.image_size = 256

        if num_classes != None:
            self.conditional = True
            self.conv1 = conv_block(channels + 12, conv_dim, use_bn=False)
        else:
            self.conv1 = conv_block(channels, conv_dim, use_bn=False)
            self.conditional = False

        self.conv2 = conv_block(conv_dim, conv_dim * 2, use_bn=False)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4, use_bn=False)
        self.conv4 = conv_block(conv_dim * 4, conv_dim * 8, use_bn=False)
        self.conv5 = conv_block(conv_dim * 8, conv_dim * 16, use_bn=False)
        self.conv6 = conv_block(conv_dim * 16, conv_dim * 32, use_bn=False)
        self.conv7 = conv_block(conv_dim * 32, 1, k_size=4, stride=4, pad=0, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label=None):
        alpha = 0.2

        if self.conditional:
            label = label.reshape([label.shape[0],12,1,1])
            #expand to image dimension
            label = label.expand([label.shape[0],12,self.image_size,self.image_size])
            x = torch.cat((x, label), dim=1)

        
        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = F.leaky_relu(self.conv4(x), alpha)
        x = F.leaky_relu(self.conv5(x), alpha)
        x = F.leaky_relu(self.conv6(x), alpha)
        x = self.conv7(x)
        return x.squeeze()