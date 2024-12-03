import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional block
# If an upscale is needed use transpose
def conv_block(c_in, c_out, k_size=4, stride=2, pad=1, use_bn=True, transpose=False):
    module = []
    if transpose:
        module.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    else:
        module.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=not use_bn))
    if use_bn:
        module.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*module)

class CNNGenerator(nn.Module):
    def __init__(self, z_dim=10, num_classes=10, label_embed_size=5, channels=3, conv_dim=64, multi_label = False, num_labels = 1):
        super(CNNGenerator, self).__init__()
        
        if num_classes != None:
            if multi_label:
                # For multilabel case we simply append the label vector to the latent vector
                self.multi_label = True
                self.conditional = True
                self.num_labels = num_labels
                self.ln = nn.Linear(z_dim + num_labels, 4*4*conv_dim*4)
            else:
                # For single label we create an embedding to convert numerical to sparse representation (more necessary for CIFAR which is not binary)
                self.label_embedding = nn.Embedding(num_classes, label_embed_size)
                self.conditional = True
                self.ln = nn.Linear(z_dim+label_embed_size, 4*4*conv_dim*4)   
        else:
            # For no conditional we simply use the latent vector
            self.ln = nn.Linear(z_dim, 4*4*conv_dim*4)
            self.conditional = False
        
        self.tconv2 = conv_block(conv_dim * 4, conv_dim * 2, transpose=True)
        self.tconv3 = conv_block(conv_dim * 2, conv_dim, transpose=True)
        self.tconv4 = conv_block(conv_dim, channels, transpose=True, use_bn=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label=None):
        if self.conditional:
            if self.multi_label:
                x = torch.cat((x, label), dim=1)
            else:
                label_embed = self.label_embedding(label)
                label_embed = label_embed.reshape([label_embed.shape[0], -1, 1, 1])
                x = torch.cat((x, label_embed), dim=1).flatten(start_dim=1)
                
        x = F.relu(self.ln(x))
        x = x.reshape(x.shape[0],-1,4,4)

        x = F.relu(self.tconv2(x))
        x = F.relu(self.tconv3(x))
        x = torch.tanh(self.tconv4(x))
        return x


class CNNDiscriminator(nn.Module):
    def __init__(self, num_classes=10, channels=3, conv_dim=64, multi_label = False, num_labels = 1):
        super(CNNDiscriminator, self).__init__()
        self.image_size = 32

        if num_classes != None:
            if multi_label:
                # For multilabel case we simply append the label vector to the input image (expanding representation below)
                self.multi_label = True
                self.conditional = True
                self.num_labels = num_labels
                self.conv1 = conv_block(channels + num_labels, conv_dim, use_bn=False)
            else:
                # For single label we create an embedding to convert numerical to sparse representation (more necessary for CIFAR which is not binary)
                self.label_embedding = nn.Embedding(num_classes, self.image_size*self.image_size)
                self.conditional = True
                self.conv1 = conv_block(channels + 1, conv_dim, use_bn=False)
        else:
            # For no conditional we simply use the image
            self.conv1 = conv_block(channels, conv_dim, use_bn=False)
            self.conditional = False

        self.conv2 = conv_block(conv_dim, conv_dim * 2, use_bn=False)
        self.conv3 = conv_block(conv_dim * 2, conv_dim * 4, use_bn=False)
        self.l4 = nn.Linear(conv_dim*4*4*4, 512)
        self.l5 = nn.Linear(512, 128)
        self.l6 = nn.Linear(128,1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, label=None):
        alpha = 0.2

        if self.conditional:
            if self.multi_label:
                label = label.reshape([label.shape[0],self.num_labels,1,1])
                #expand to image dimension
                label = label.expand([label.shape[0],self.num_labels,self.image_size,self.image_size])
                x = torch.cat((x, label), dim=1)
            else:
                label_embed = self.label_embedding(label)
                label_embed = label_embed.reshape([label_embed.shape[0], 1, self.image_size, self.image_size])
                x = torch.cat((x, label_embed), dim=1)

        x = F.leaky_relu(self.conv1(x), alpha)
        x = F.leaky_relu(self.conv2(x), alpha)
        x = F.leaky_relu(self.conv3(x), alpha)
        x = self.l4(x.flatten(start_dim=1))
        x = self.l5(x)
        x = self.l6(x)
        return x.squeeze()     