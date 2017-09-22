import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from pyramidnet import *
import vgg
from inceptionresnetv2.pytorch_load import inceptionresnetv2


class AmazonModel(nn.Module):
    def __init__(self):
        super(AmazonModel,self).__init__()
        print('Loading ResNet...')
        self.resnet34 = models.resnet34(pretrained=True)
        self.fc1 = nn.Linear(1000,512)
        self.fc2 = nn.Linear(512,17)
        print('Resnet Loaded')

    def forward(self, x):
        x = self.resnet34(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        return x

    def getname(self):
        return 'AmazonModel'


class AM_alex(nn.Module):
    def __init__(self):
        super(AM_alex,self).__init__()
        self.net = models.alexnet(pretrained=True)
        self.fc = nn.Linear(1000,17)

    def forward(self, x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'alex'

class AM_resnet101(nn.Module):
    def __init__(self):
        super(AM_resnet101,self).__init__()
        self.net = models.resnet101(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1000,17)

    def forward(self, x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet101'

class AM_resnet101_new(nn.Module):
    def __init__(self):
        super(AM_resnet101_new,self).__init__()
        self.net = models.resnet101(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(2)
        self.net.fc = nn.Linear(4*2048,1000)
        self.fc = nn.Linear(1000,17)

    def forward(self, x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet101_new'

class AM_resnet34(nn.Module):
    def __init__(self):
        super(AM_resnet34, self).__init__()
        self.net = models.resnet34(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1000, 17)

    def forward(self, x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet34'

class AM_resnet34_new(nn.Module):
    def __init__(self):
        super(AM_resnet34_new, self).__init__()
        self.net = models.resnet34(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(2)
        self.net.fc = nn.Linear(512*4,1000)
        self.fc = nn.Linear(1000, 17)

    def forward(self, x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet34_new'

class AM_inception(nn.Module):
    def __init__(self):
        super(AM_inception,self).__init__()
        self.net = inceptionresnetv2(pretrained=True)
        self.net.classif = nn.Linear(1536,17)

    def forward(self,x):
        x = self.net(x)
        return F.sigmoid(x)

    def getname(self):
        return 'inception'

class AM_squeezenet11(nn.Module):
    def __init__(self):
        super(AM_squeezenet11,self).__init__()
        self.net = models.squeezenet1_1(pretrained=True)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'squeezenet11'

class AM_vgg13(nn.Module):
    def __init__(self):
        super(AM_vgg13,self).__init__()
        self.net = vgg.vgg13_bn(pretrained=True)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'vgg13'

class AM_resnet152(nn.Module):
    def __init__(self):
        super(AM_resnet152,self).__init__()
        self.net = models.resnet152(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet152'

class AM_resnet50(nn.Module):
    def __init__(self):
        super(AM_resnet50,self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet50'

class AM_resnet50_new(nn.Module):
    def __init__(self):
        super(AM_resnet50_new,self).__init__()
        self.net = models.resnet50(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(2)
        self.net.fc = nn.Linear(4*2048,1000)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet50_new'

class AM_densenet121(nn.Module):
    def __init__(self):
        super(AM_densenet121,self).__init__()
        self.net = models.densenet121(pretrained=True)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.densenet121_forward(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def densenet121_forward(self,x):
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out,1).view(features.size(0), -1)
        out = self.net.classifier(out)
        return out

    def getname(self):
        return 'densenet121'

class AM_resnet18(nn.Module):
    def __init__(self):
        super(AM_resnet18,self).__init__()
        self.net = models.resnet18(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet18'

class AM_resnet18_new(nn.Module):
    def __init__(self):
        super(AM_resnet18_new,self).__init__()
        self.net = models.resnet18(pretrained=True)
        self.net.avgpool = nn.AdaptiveAvgPool2d(2)
        self.net.fc = nn.Linear(512*4,1000)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def getname(self):
        return 'resnet18_new'

class AM_densenet161(nn.Module):
    def __init__(self):
        super(AM_densenet161,self).__init__()
        self.net = models.densenet161(pretrained=True)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.densenet161_forward(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def densenet161_forward(self,x):
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        print(out.size())
        out = F.avg_pool2d(out,7).view(features.size(0), -1)
        print(out.size())
        out = self.net.classifier(out)
        return out

    def getname(self):
        return 'densenet161'

class AM_densenet161_new(nn.Module):
    def __init__(self):
        super(AM_densenet161_new,self).__init__()
        self.net = models.densenet161(pretrained=True)
        self.net.classifier = nn.Linear(2208*4,1000)
        self.fc = nn.Linear(1000,17)
        setattr(self.net.features,'pool0',nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

    def forward(self,x):
        x = F.relu(self.densenet161_forward(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def densenet161_forward(self,x):
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out,2).view(features.size(0), -1)
        out = self.net.classifier(out)
        return out

    def getname(self):
        return 'densenet161_new'

class AM_densenet169_new(nn.Module):
    def __init__(self):
        super(AM_densenet169_new,self).__init__()
        self.net = models.densenet169(pretrained=True)
        self.net.classifier = nn.Linear(4*1664,1000)
        self.fc = nn.Linear(1000,17)
        setattr(self.net.features,'pool0',nn.MaxPool2d(kernel_size=2,stride=2,padding=1))

    def forward(self,x):
        x = F.relu(self.densenet_forward(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def densenet_forward(self,x):
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out,2).view(features.size(0), -1)
        out = self.net.classifier(out)
        return out

    def getname(self):
        return 'densenet169_new'

class AM_densenet201_new(nn.Module):
    def __init__(self):
        super(AM_densenet201_new,self).__init__()
        self.net = models.densenet201(pretrained=True)
        self.net.classifier = nn.Linear(4*1920,1000)
        self.fc = nn.Linear(1000,17)
        setattr(self.net.features,'pool0',nn.MaxPool2d(kernel_size=2,stride=2,padding=1))

    def forward(self,x):
        x = F.relu(self.densenet_forward(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def densenet_forward(self,x):
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out,2).view(features.size(0), -1)
        out = self.net.classifier(out)
        return out

    def getname(self):
        return 'densenet201_new'

class AM_densenet121_new(nn.Module):
    def __init__(self):
        super(AM_densenet121_new,self).__init__()
        self.net = models.densenet121(pretrained=True)
        self.net.classifier = nn.Linear(4*1024,1000)
        self.fc = nn.Linear(1000,17)
        setattr(self.net.features,'pool0',nn.MaxPool2d(kernel_size=2,stride=2,padding=1))

    def forward(self,x):
        x = F.relu(self.densenet_forward(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def densenet_forward(self,x):
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out,2).view(features.size(0), -1)
        out = self.net.classifier(out)
        return out

    def getname(self):
        return 'densenet121_new'

class AM_densenet161_newnew(nn.Module):
    def __init__(self):
        super(AM_densenet161_newnew,self).__init__()
        self.net = models.densenet161(pretrained=True)
        self.net.classifier = nn.Linear(2208*64,2208*32)
        self.fc0_0 = nn.Linear(2208*32,2208*8)
        self.fc0_1 = nn.Linear(2208*8,1000)
        self.fc = nn.Linear(1000,17)
        setattr(self.net.features,'pool0',nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

    def forward(self,x):
        x = F.relu(self.densenet161_forward(x))
        x = F.dropout(x,0.5)
        x = F.relu(self.fc0_0(x))
        x = F.relu(self.fc0_1(x))
        x = self.fc(x)
        return F.sigmoid(x)

    def densenet161_forward(self,x):
        features = self.net.features(x)
        out = F.relu(features, inplace=True)
        out = out.view(features.size(0), -1)
        out = self.net.classifier(out)
        return out

    def getname(self):
        return 'densenet161_newnew'

class AM_vgg11(nn.Module):
    def __init__(self):
        super(AM_vgg11,self).__init__()
        self.net = models.vgg11(pretrained=True)
        self.fc = nn.Linear(1000,17)

    def forward(self,x):
        x = F.relu(self.net(x))
        x = self.fc(x)
        return F.sigmoid(x)


    def getname(self):
        return 'vgg11'

if __name__=='__main__':
    # m = AM_densenet161_newnew()
    # m = models.resnet34(pretrained=True)
    # m = AM_resnet34_new()
    # m = models.resnet50(pretrained=True)
    # m = AM_resnet50_new()
    # m = models.resnet101(pretrained=True)
    # m = AM_resnet101_new()
    # m = AM_resnet18_new()
    m = AM_vgg11()
    # m = models.densenet121(pretrained=True)
    x = Variable(torch.randn((3,3,224,224)))
    # m = models.densenet121(pretrained=True)
    # setattr(m.features,)
    # m = models.densenet169(pretrained=True)
    y = m(x)
    print(y)

    # y = m(x)
    # models.re
    # print(y)