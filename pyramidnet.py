# [1] "Feature Pyramid Networks for Object Detection" -  Tsung-Yi Lin, Piotr Dollár,
#         Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie, arxiv 2016
#         https://arxiv.org/abs/1612.03144
#
# [2] "DSSD : Deconvolutional Single Shot Detector" - Cheng-Yang Fu, Wei Liu, Ananth Ranga,
#         Ambrish Tyagi, Alexander C. Berg, arxiv 2017
#
# [3] "Aggregated Residual Transformations for Deep Neural Networks" - Saining Xie,
#         Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He, arxiv 2016
#         https://github.com/D-X-Y/ResNeXt/blob/master/models/resnext.py
#
# [4] "Is object localization for free? – Weakly-supervised learning with convolutional neural networks" -
#         Maxime Oquab, Léon Bottou, Ivan Laptev, Josef Sivic, cvpr 2015
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

# from net.common import *
# from net.utility.tool import *



#----- helper functions --------

def make_linear_bn_prelu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.PReLU(out_channels),
    ]


def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_linear_bn_relu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
    ]


def make_max_flat(out):
    flat = F.adaptive_max_pool2d(out,output_size=1)  ##nn.AdaptiveMaxPool2d(1)(out)
    flat = flat.view(flat.size(0), -1)
    return flat


def make_avg_flat(out):
    flat =  F.adaptive_avg_pool2d(out,output_size=1)
    flat = flat.view(flat.size(0), -1)
    return flat


def make_shortcut(out, modifier):
    if modifier is None:
        return out
    else:
        return modifier(out)

def make_flat(out):
    #flat =  F.adaptive_avg_pool2d(out,output_size=4)
    out  = F.avg_pool2d(out,kernel_size=4, stride=2, padding=0)
    out  = F.adaptive_max_pool2d(out,output_size=1)
    flat = out.view(out.size(0), -1)
    return flat


#############################################################################3


class PyNet_10(nn.Module):

    def __init__(self, in_shape, num_classes):
        super(PyNet_10, self).__init__()
        in_channels, height, width = in_shape

        self.preprocess = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(16, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(16, 16, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(16, 16, kernel_size=1, stride=1, padding=0 ),
        ) # 128

        self.conv1d = nn.Sequential(
            *make_conv_bn_relu(16,32, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32,64, kernel_size=1, stride=1, padding=0 ),
        ) # 128
        self.shortld = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0, bias=False)


        self.conv2d = nn.Sequential(
            *make_conv_bn_relu(64,64,  kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(64,64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64,128, kernel_size=1, stride=1, padding=0 ),
        ) # 64
        self.short2d = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv3d = nn.Sequential(
            *make_conv_bn_relu(128,128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(128,128, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(128,256, kernel_size=1, stride=1, padding=0 ),
        ) # 32
        self.short3d = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv4d = nn.Sequential(
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(256,256, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
        ) # 16
        self.short4d = None #nn.Identity()

        self.conv5d = nn.Sequential(
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(256,256, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
        ) # 8
        self.short5d = None #  nn.Identity()


        self.conv4u = nn.Sequential(
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(256,256, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(256,256, kernel_size=1, stride=1, padding=0 ),
        ) # 16

        self.conv3u = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(128,128, kernel_size=3, stride=1, padding=1, groups=16 ),
            *make_conv_bn_relu(128,128, kernel_size=1, stride=1, padding=0 ),
        ) # 32

        self.conv2u = nn.Sequential(
            *make_conv_bn_relu(128,64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 64,64, kernel_size=1, stride=1, padding=0 ),
        ) # 64

        self.conv1u = nn.Sequential(
            *make_conv_bn_relu(64,64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64,64, kernel_size=1, stride=1, padding=0 ),
        ) # 128




        self.cls2d = nn.Sequential(
            *make_linear_bn_relu(128, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls3d = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls4d = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls5d = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )

        self.cls1u = nn.Sequential(
            *make_linear_bn_relu(64,  512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls2u = nn.Sequential(
            *make_linear_bn_relu( 64, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls3u = nn.Sequential(
            *make_linear_bn_relu(128, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )
        self.cls4u = nn.Sequential(
            *make_linear_bn_relu(256, 512),
            *make_linear_bn_relu(512, 512),
            nn.Linear(512, num_classes)
        )



    def forward(self, x):

        out    = self.preprocess(x)                                       #128

        conv1d = self.conv1d(out)                                         #128
        out    = F.max_pool2d(conv1d, kernel_size=2, stride=2)  # 64

        conv2d = self.conv2d(out) + make_shortcut(out, self.short2d)      # 64
        out    = F.max_pool2d(conv2d, kernel_size=2, stride=2) # 32
        flat2d = make_max_flat(out)

        conv3d = self.conv3d(out) + make_shortcut(out, self.short3d)      # 32
        out    = F.max_pool2d(conv3d, kernel_size=2, stride=2) # 16
        flat3d = make_max_flat(out)

        conv4d = self.conv4d(out) + make_shortcut(out, self.short4d)      # 16
        out    = F.max_pool2d(conv4d, kernel_size=2, stride=2) #  8
        flat4d = make_max_flat(out)

        conv5d = self.conv5d(out) + make_shortcut(out, self.short5d)      #  8
        out    = conv5d                                        #  4
        flat5d = make_max_flat(out)

        out    = F.upsample_bilinear(out,scale_factor=2)      # 16
        out    = out + conv4d
        out    = self.conv4u(out)
        flat4u = make_max_flat(out)

        out    = F.upsample_bilinear(out,scale_factor=2)      # 32
        out    = out + conv3d
        out    = self.conv3u(out)
        flat3u = make_max_flat(out)

        out    = F.upsample_bilinear(out,scale_factor=2)      # 64
        out    = out + conv2d
        out    = self.conv2u(out)
        flat2u = make_max_flat(out)

        out    = F.upsample_bilinear(out,scale_factor=2)      #128
        out    = out + conv1d
        out    = self.conv1u(out)
        flat1u = make_max_flat(out)



        logit2d = self.cls2d(flat2d).unsqueeze(2)
        logit3d = self.cls3d(flat3d).unsqueeze(2)
        logit4d = self.cls4d(flat4d).unsqueeze(2)
        logit5d = self.cls5d(flat5d).unsqueeze(2)

        logit1u = self.cls1u(flat1u).unsqueeze(2)
        logit2u = self.cls2u(flat2u).unsqueeze(2)
        logit3u = self.cls3u(flat3u).unsqueeze(2)
        logit4u = self.cls4u(flat4u).unsqueeze(2)


        logit = torch.cat((logit2d,logit3d,logit4d,logit5d,logit1u,logit2u,logit3u,logit4u),2)

        logit = F.dropout(logit, p=0.15,training=self.training)
        logit = logit.sum(2)
        logit = logit.view(logit.size(0),logit.size(1)) #unsqueeze(2)
        prob = F.sigmoid(logit)

        return prob

    def getname(self):
        return 'pynet10'


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    #inputs = torch.randn(96,3,128,128)
    inputs = torch.randn(1,3,112,112)
    #inputs = torch.randn(96,3,96,96)

    in_shape = inputs.size()[1:]
    num_classes = 17

    if 1:
        net = PyNet_10(in_shape,num_classes).cuda().train()
        x = Variable(inputs).cuda()

        start = timer()
        logit,prob = net.forward(x)
        end = timer()
        print ('cuda(): end-start=%0.0f  ms'%((end - start)*1000))

        #dot = make_dot(y)
        #dot.view()
        print(type(net))
        print(net)
        print(prob)