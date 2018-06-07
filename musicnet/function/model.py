import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import numpy as np
sys.path.append('../fun')
import math

class block(nn.Module):
    def __init__(self, inp, out):
        super(block, self).__init__()
        self.bn1 = nn.BatchNorm2d(inp)       
        self.conv1 = nn.Conv2d(inp, out, (3,1), padding=(1,0))
        self.bn2 = nn.BatchNorm2d(out)       
        self.conv2 = nn.Conv2d(out, out, (3,1), padding=(1,0))
        self.bn3 = nn.BatchNorm2d(out)       

        self.sk = nn.Conv2d(inp, out, (1,1), padding=(0,0))
    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn3(out)
        out += self.sk(x)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dp = nn.Dropout(.0)
        
        def SY_model():
            fs = (3,1)
            ps = (1,0)
            fre = 88
            inp = fre
            num_labels = 7
            self.head = nn.Sequential(
                nn.BatchNorm2d(inp),       
                nn.Conv2d(inp, fre, (3,1), padding=(1,0)),
                block(fre, fre*2),
                nn.Dropout(p=0.2),
                nn.MaxPool2d((3,1),(3,1)),
                block(fre*2, fre*3),
                nn.Dropout(p=0.2),
                nn.MaxPool2d((3,1),(3,1)),
                block(fre*3, fre*3),
                nn.BatchNorm2d(fre*3),
                nn.ReLU(inplace=True),
                nn.Conv2d( fre*3, num_labels, fs, padding=ps)
            )

    
        SY_model()

    def forward(self, _input, Xavg, Xstd):

        def get_x(x,avg,std):
            xs = x.size()
            avg = avg.view(1, avg.size()[0],1,1).repeat(xs[0], 1, xs[2], xs[3]).type('torch.cuda.FloatTensor')
            std = std.view(1, std.size()[0],1,1).repeat(xs[0], 1, xs[2], xs[3]).type('torch.cuda.FloatTensor')
            x = (x - avg)/std
            return x
        x = _input
        x = get_x(x,Xavg,Xstd)
        frame_pred = self.head(x)
        
        return frame_pred

        

        
        