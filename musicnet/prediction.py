#!/usr/bin/python
# -*- coding: UTF-8 -*-
import sys
import os
import torch.optim as optim
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
import torch.nn.init as init
from torch.utils.data import Dataset
sys.path.append('./function')
from model import * 
date = datetime.datetime.now()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def norm(avg, std, data, size):
    avg = np.tile(avg.reshape((1,-1,1,1)),(size[0],1,size[2],size[3]))
    std = np.tile(std.reshape((1,-1,1,1)),(size[0],1,size[2],size[3]))
    data = (data - avg)/std
    return data
    
def load_te_mp3(name,avg, std):
    sr = 44100
    chunk_size = sr*3/512
    x = [] #change
    h_size = 9
    #ex_spectrogram
    data, sr = librosa.load('mp3/'+name,sr=sr)
    cqt = librosa.cqt(data, sr=sr, hop_length=512, fmin=27.5*float(1), n_bins=88, bins_per_octave=12)
    cqt = ((1.0/80.0) * librosa.core.amplitude_to_db(np.abs(cqt), ref=np.max)) + 1.0
    #chunk data
    cqt = cqt[:,:int(cqt.shape[1]/chunk_size)*chunk_size]
    for i in range(int(cqt.shape[1]/chunk_size)):
        data=cqt[:,i*chunk_size:i*chunk_size+chunk_size]
        x.append(data)
    x = np.array(x) 

    x = np.expand_dims(x, axis=3)
    #tesize = x.shape
    #x = norm(avg, std, Xte, tesize)

    return x

def model_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif classname.find('Linear') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
       # init.constant(m.bias, 0)

# Dataset
class Data2Torch(Dataset):
    def __init__(self, data):
        self.X = data[0]

    def __getitem__(self, index):
        mX = torch.from_numpy(self.X[index]).float()
        return mX
    
    def __len__(self):
        return len(self.X)

def main(argv):
    #load model
    model = Net().cuda()
    model.apply(model_init)
    model_dict = model.state_dict()
    save_dic = torch.load('./data/params') 
    va_th = save_dic['va_th']
    pretrained_dict1 = {k: v for k, v in save_dic['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict1) 
    model.load_state_dict(model_dict)
    print 'finishing loading model'

    name = argv[0]
    #load test dataset
    Xavg, Xstd = save_dic['avg'], save_dic['std']
    Xte = load_te_mp3(name,Xavg, Xstd)
    print 'finishing loading dataset'

    #predict configure
    v_kwargs = {'batch_size': 8, 'num_workers': 10, 'pin_memory': True}
    loader = torch.utils.data.DataLoader(Data2Torch([Xte]), **v_kwargs)

    all_pred = np.zeros((Xte.shape[0],7,int(Xte.shape[2]/9)))

    #start predict
    print 'start predicting...'
    model.eval()
    ds = 0
    for idx,_input in enumerate(loader):
        data = Variable(_input.cuda())
        pred = model(data, Xavg, Xstd)
        all_pred[ds: ds + len(data)] = F.sigmoid(torch.squeeze(pred)).data.cpu().numpy()
        ds += len(data)

    for i, (p) in enumerate(all_pred):
        p = p - np.expand_dims(va_th,1)
        p[p>0] = 1
        p[p<0] = 0
        if i == 0: pre = p
        else: pre = np.append(pre, p, axis=1)
    pre = np.vstack(pre)
    np.save('result/'+name[:-3]+'npy',pre)
    plt.figure(figsize=(10,3))
    plt.yticks(np.arange(8), ('Piano', 'Violin', 'Viola', 'Cello', 'Clarinet', 'Bassoon', 'Horn'))
    plt.imshow(pre,cmap=plt.cm.binary, interpolation='nearest', aspect='auto')
    plt.savefig('plot/'+name+'.png')
    print 'finish!'

if __name__ == "__main__":
   main(sys.argv[1:])

	
