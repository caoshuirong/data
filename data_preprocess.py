#!/usr/bin/env python
# coding: utf-8

# In[443]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all" 


# ## 1.1 生成数据集 （修改window_len即可）

# In[444]:

from tqdm import tqdm
import pandas as pd
import numpy as np
from itertools import product
import os
import sys
if not os.path.isdir('./checkpoint'):
    os.mkdir('./checkpoint')
if not os.path.isdir('./output'):
    os.mkdir('./output')
# In[445]:


data = pd.read_csv('train.csv',header = None)

# data = data.to_numpy()
total_num = data.shape[0]
total_len = data.shape[1]
total_num
total_len

output_size = 56
window_len = 56
interval = 30
# In[446]:


# max_min 归一化
def max_min(x,total_num):
    min_val = x.values.min(axis=1)
    max_val = x.values.max(axis=1)
    values = (x.values - min_val.reshape(total_num,1))/(max_val - min_val).reshape(total_num,1)

    return values,min_val,max_val

def mean_std(x,total_num):
    mean = x.values.mean(axis=1)
    std = x.values.std(axis=1)
    values = (x.values - mean.reshape(total_num,1))/std.reshape(total_num,1)

    return values,mean,std
# In[447]:


# 最后生成测试数据时进行还原，然后与测试的label进行比较
def max_min_reverse(x_numpy,min_val,max_val):
    values = (x_numpy + min_val) * (max_val - min_val)
    return values

# In[448]:



# max_min 归一化
data,min_val,max_val = max_min(data,total_num)

# data = max_min_reverse(data,min_val,max_val)


# In[449]:


data[0,:]


# In[450]:


# 划分训练集和验证集



train_series = data[:,0:total_len-output_size]
train_size = train_series.shape[1]

y_val_list = [data[i,total_len-output_size:] for i in range(total_num)]
y_val_list[0].shape

x_test_list = [data[i,total_len-window_len:] for i in range(total_num)]
x_test_list[0].shape


# In[451]:


# todo 滑动窗口截取
# 每个样本的shape为(111,window_len) (111,window_len:window_len+56)


x_train_list = []
y_train_list = []
x_val_list = []

x_train_max_val=[]
x_train_min_val=[]
for i in range(total_num):
    for j in range(0,train_size - window_len-output_size,interval):
        x_train = train_series[i,j:j+window_len]
        y_train = train_series[i,j+window_len:j+window_len+output_size]
        x_train_list.append(x_train)
        y_train_list.append(y_train)
        x_train_max_val.append(max_val[i])
        x_train_min_val.append(min_val[i])

x_val_list = [train_series[i,train_size-window_len:] for i in range(total_num)]

dataset ={
    'x_train':x_train_list,
    'y_train':y_train_list,
    'x_train_max_val':x_train_max_val,
    'x_train_min_val':x_train_min_val,
    'x_val':x_val_list,
    'y_val':y_val_list,
    'x_test':x_test_list,
    'max_val':max_val,
    'min_val':min_val
}

# todo 问题在于 700多维的数据，LSTM记不住该如何划分。


# In[452]:


import pickle
with open('dataset_%d.pkl'%window_len,'wb') as f:
    pickle.dump(dataset,f)


# ## 1.2 封装数据

# In[453]:


import torch
from torch import nn
import torch.functional as f
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim


# In[454]:


class MyDataset(Dataset):
    def __init__(self,x_train,y_train,x_train_max_val,x_train_min_val):
        super(Dataset,self).__init__()
        self.length = len(x_train)
        self.data = [torch.Tensor(x_train[i]) for i in range(self.length)]
        self.label = [torch.Tensor(y_train[i]) for i in range(self.length)]
        self.x_train_max_val = [torch.Tensor([x_train_max_val[i]]) for i in range(self.length)]
        self.x_train_min_val = [torch.Tensor([x_train_min_val[i]]) for i in range(self.length)]
    
    def __getitem__(self,index):
        return self.data[index],self.label[index],self.x_train_max_val[index],self.x_train_min_val[index]
    
    def __len__(self):
        return self.length


# In[455]:


# 封装训练和验证集
x_train = dataset['x_train']
y_train = dataset['y_train']
x_train_max_val = dataset['x_train_max_val']
x_train_min_val = dataset['x_train_min_val']
x_val = dataset['x_val']
y_val = dataset['y_val']
min_val = dataset['min_val']
max_val = dataset['max_val']
    

train_dataset = MyDataset(x_train,y_train,x_train_max_val,x_train_min_val)
val_dataset = MyDataset(x_val,y_val,max_val,min_val)

train_loader = DataLoader(train_dataset,batch_size = 32,shuffle = True)
val_loader = DataLoader(val_dataset,batch_size = total_num)


# In[456]:


# 封装测试集
x_test = dataset['x_test']
y_test = pd.read_csv('test.csv',header = None)
y_test = [y_test.iloc[i,:].to_numpy() for i in range(len(x_test))]
test_dataset = MyDataset(x_test,y_test,max_val,min_val)
test_loader = DataLoader(test_dataset,batch_size = total_num)


# # 3.搭建模型

# In[457]:


import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# In[458]:


from enum import Enum
class LossFunction(Enum):
    MAE = 1
    MSE = 2
    SmoothL1 = 3

class Mode(Enum):
    Trian = 1
    Valid = 2
    Test = 3

class Optim(Enum):
    SGD = 1
    Adam = 2

class ModelType(Enum):
    LSTM = 1
    MLP = 2

# In[459]:


def initial_weights(model):
    for m in model.modules():
#         if isinstance(m, nn.LSTM):
#             [nn.init.orthogonal_(para) for name,para in m.name_parameters() if 'weight' in name]
                
        if isinstance(m,nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

class MLP(nn.Module):
    def __init__(self,input_size,output_size,loss_type=LossFunction.MAE,dropout_rate = 0.2):
        super(MLP, self).__init__()
        self.name = 'MLP'
        self.loss_type_name = loss_type.name
        self.model = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256,output_size),
            nn.ReLU()
        )
        if loss_type == LossFunction.MAE:
            self.loss_fun = nn.L1Loss()
        elif loss_type == LossFunction.MSE:
            self.loss_fun = nn.MSELoss()
        elif loss_type == LossFunction.SmoothL1:
            self.loss_fun = nn.SmoothL1Loss()
        else:
            raise ValueError("please check loss_fun type!")
        self.apply(initial_weights)

    def forward(self,x,label):
        out = self.model(x)
        if label is not None:
            loss = self.loss_fun(out,label)
            out = (out,loss)
        return out


class Encoder(nn.Module):
    def __init__(self,input_size,rnn_inpt_size,rnn_hidden_size,dropout_rate=0.2):
        super(Encoder,self).__init__()
        self.input_size = input_size
        self.rnn_input_size = rnn_inpt_size
        self.rnn_hidden_size = rnn_hidden_size
        
        self.linear = nn.Linear(input_size,rnn_inpt_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm = nn.LSTM(rnn_inpt_size,rnn_hidden_size,batch_first=True)
        self.apply(initial_weights)
    
    def forward(self,x):
        emb = self.dropout(self.linear(x)) # batch size * seq_len
        _,(h_n,_) = self.lstm(emb.reshape(x.shape[0],1,self.rnn_input_size))
        
        return h_n
        
    


# In[460]:


class Decoder(nn.Module):
    def __init__(self,ouput_size,rnn_inpt_size,rnn_hidden_size,dropout_rate=0.2):
        super(Decoder,self).__init__()
        self.output_size = output_size
        self.rnn_input_size = rnn_inpt_size
        self.rnn_hidden_size = rnn_hidden_size
        
        self.linear = nn.Linear(rnn_hidden_size,ouput_size)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.lstm = nn.LSTM(rnn_inpt_size,rnn_hidden_size,batch_first=True)
#         self.apply(initial_weights)
    
    def forward(self,x):
        emb = self.linear(x.reshape(-1,self.rnn_hidden_size)) # batch size * seq_len
        
        return emb
        


# In[461]:


class Seq2Seq(nn.Module):
    def __init__(self,input_size,output_size,rnn_inpt_size,rnn_hidden_size,loss_type=LossFunction.MAE,dropout_rate=0.2):
        super(Seq2Seq,self).__init__()
        self.name = 'LSTM'
        self.loss_type_name = loss_type.name
        self.encoder = Encoder(input_size,rnn_inpt_size,rnn_hidden_size,dropout_rate)
        self.decoder = Decoder(output_size,rnn_hidden_size,rnn_hidden_size,dropout_rate)
#         self.min = min_val
#         self.max = max_val
        if loss_type == LossFunction.MAE:
            self.loss_fun = nn.L1Loss()
        elif loss_type == LossFunction.MSE:
            self.loss_fun = nn.MSELoss()
        elif loss_type == LossFunction.SmoothL1:
            self.loss_fun = nn.SmoothL1Loss()
        else:
            raise ValueError("please check loss_fun type!")
        
    def forward(self,x,label=None):
        out = self.encoder(x)
        out = self.decoder(out)
#         if mode == Mode.Test:
#             out = max_min_reverse(out.cpu().detach().numpy(),self.min,self.max)
        if label is not None:
            loss = self.loss_fun(out,label)
            out = (out,loss)
            
        return  out

    


# # 4.训练模型

# In[462]:


from matplotlib import pyplot as plt


# In[463]:


def smape(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true) / (torch.abs((y_pred + y_true)/2))) * 100


# In[464]:



class Trainer:
    def __init__(self,model,train_loader,val_loader,test_loader,train_args):
        super(Trainer,self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = train_args['lr']
        self.batch_size = train_args['batch_size']
        self.optim_type = train_args['optim']
        self.epochs = train_args['epochs']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.model.to(self.device)
        self.min_loss = 1e8
        self.best_model = None

        if self.optim_type == Optim.SGD:
            self.optim = optim.SGD(self.model.parameters(),lr = self.lr)
        elif self.optim_type == Optim.Adam:
            self.optim = optim.Adam(self.model.parameters(),lr = self.lr)
        else:
            raise ValueError("Not a recognized optimizer")
    
    def __train_one_peoch(self):
        self.model.train()
        train_loss = []
        train_smape = []
        for idx,(x_batch,y_batch,x_max_val,x_min_val) in enumerate(train_loader):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            pred,loss = self.model(x_batch,y_batch)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            train_loss.append(loss.item())
            train_smape.append(self.SMAPE(pred,y_batch,x_min_val,x_max_val))
        
        self.model.eval()
        with torch.no_grad():
            val_loss = []
            val_smape = []
            for idx,(x_batch,y_batch,x_max_val,x_min_val) in enumerate(val_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred,loss = self.model(x_batch,y_batch)
                val_loss.append(loss.item())
                val_smape.append(self.SMAPE(pred,y_batch,x_min_val,x_max_val))
            epoch_train_loss = sum(train_loss) / len(train_loss)
            epoch_train_smape = sum(train_smape) / len(train_smape)
            epoch_val_loss = sum(val_loss) / len(val_loss)
            epoch_val_smape = sum(val_smape) / len(val_smape)
            
        if epoch_val_loss < self.min_loss:
            self.min_loss = epoch_val_loss
            self.best_model = self.model

        return epoch_train_loss,epoch_val_loss,epoch_train_smape,epoch_val_smape
    
    
    def train(self):
        train_loss_list = []
        val_loss_list = []
        epoch_train_smape_list = []
        epoch_val_smape_list = []
        info = ' '
        processbar = tqdm(range(self.epochs))

        for i in range(self.epochs):
            train_loss,val_loss,epoch_train_smape,epoch_val_smape = self.__train_one_peoch()
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            epoch_train_smape_list.append(epoch_train_smape)
            epoch_val_smape_list.append(epoch_val_smape)
            if i % 21 == 0:
                self.save_model(i)
            if i % 10 == 0:
                info_str = "train_loss:%.4f val_loss:%.4f train_smape:%.4f val_smape:%.4f\n" % (train_loss, val_loss, epoch_train_smape, epoch_val_smape)
                print(info_str)
            processbar.update(1)
        title = 'len_%d + %s + %s + %s'%(window_len,self.best_model.name,self.best_model.loss_type_name,self.optim_type.name)
        info_str = title + ' ' + info_str
        logging.info(info_str)
        self.draw(train_loss_list,val_loss_list,'epochs','loss',['train_loss','val_loss'],title)
        self.draw(epoch_train_smape_list,epoch_val_smape_list,'epochs','smape',['train_smape','val_smape'],title)
        return min(train_loss_list),min(val_loss_list),min(epoch_train_smape_list).item(),min(epoch_val_smape_list).item()
        
        
    def draw(self,y1,y2,xlabel,ylabel,legend,title):
        x = [i for i in range(len(y1))]
        plt.plot(x,y1,x,y2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
        plt.title(title)
        plt.savefig("./output/%s.png" % (title +'_'+ ylabel))
        plt.show()
        
        
    def eval(self,test_loader):
        self.model.eval()
        test_loss = []
        with torch.no_grad():
            # batch_size == total_num
            for idx,(x_batch,y_batch,x_max_val,x_min_val) in enumerate(test_loader):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred,_= self.model(x_batch,y_batch)
            test_loss = self.loss_function(pred,y_batch,x_min_val,x_max_val)
            test_smape = self.SMAPE(pred,y_batch,x_min_val,x_max_val,Mode.Test)
            test_info = 'test_loss : %.6f test_smape : %.6f' % (test_loss,test_smape)
            logging.info(test_info)
        return test_loss,test_smape

    
    def SMAPE(self,pred,y_batch,normalize_min_coef,normalize_max_coef,mode=Mode.Trian):
        pred = max_min_reverse(pred.cpu(),normalize_min_coef,normalize_max_coef)
        if mode == Mode.Test:
            y_batch = y_batch.cpu()
        else:
            y_batch = max_min_reverse(y_batch.cpu(),normalize_min_coef,normalize_max_coef)
        smape_val= smape(y_batch.reshape(-1),pred.reshape(-1))
        return smape_val

    def loss_function(self,pred,y_batch,normalize_min_coef,normalize_max_coef):
        if self.model.loss_type_name == LossFunction.MAE.name:
            loss_fun = nn.L1Loss()
        elif self.model.loss_type_name == LossFunction.MSE.name:
            loss_fun = nn.MSELoss()
        elif self.model.loss_type_name == LossFunction.SmoothL1.name:
            loss_fun = nn.SmoothL1Loss()
        tmp,min,max = y_batch.cpu(), normalize_min_coef, normalize_max_coef
        values = (tmp - min.reshape(total_num, 1)) / (max - min).reshape(total_num, 1)
        loss = loss_fun(pred,torch.Tensor(values))
        return loss

    def save_model(self,epoch=None):
        if epoch is not None:
            torch.save(self.best_model,'./checkpoint/len=%d + %s + %s + %s + checkpoint: %d.pt'
                       %(window_len,self.best_model.name,self.best_model.loss_type_name,self.optim_type.name,epoch))
        torch.save(self.best_model,'./checkpoint/len=%d + %s + %s + %s + test.pt'
                       %(window_len,self.best_model.name,self.best_model.loss_type_name,self.optim_type.name))


# In[465]:

if __name__=='__main__':

    loss_fun_list = [LossFunction.MAE, LossFunction.MSE, LossFunction.SmoothL1]
    optim_list = [Optim.SGD, Optim.Adam]
    combination = product(loss_fun_list,optim_list)

    recoder = pd.DataFrame()

    best_train_loss_list = []
    best_valid_loss_list = []
    best_train_smape_list = []
    best_valid_smape_list = []
    best_test_loss_list = []
    best_test_smape_list = []

    for loss_type,optim_type in combination:
        model_args = {
            "input_size": window_len,
            'output_size':output_size,
            "rnn_inpt_size" : 128,
            "rnn_hidden_size": 256,
            "loss_type":loss_type,
            'dropout_rate':0.2,
            'model_type':ModelType.LSTM
        }
        lr = 6e-3 if optim_type == Optim.SGD else 3e-4
        train_args = {
            'lr' :lr,
            'batch_size' : 32,
            'dropout_rate' : 0.5,
            'optim' : optim_type,
            'epochs' : 50,
        }

        if model_args['model_type'] == ModelType.LSTM:
            model = Seq2Seq(input_size=model_args["input_size"],output_size=model_args['output_size'],
                        rnn_inpt_size=model_args['rnn_inpt_size'],rnn_hidden_size=model_args['rnn_hidden_size'],
                        loss_type=model_args['loss_type'],dropout_rate=model_args['dropout_rate'])
        elif model_args['model_type'] == ModelType.MLP:
            model = MLP(model_args["input_size"],model_args['output_size'])

        trainer = Trainer(model,train_loader,val_loader,test_loader,train_args)


        # In[466]:

        best_train_loss, best_valid_loss, best_train_smape, best_valid_smape = trainer.train()
        best_test_loss, best_test_smape = trainer.eval(test_loader)
        print('test_loss : %.6f test_smape : %.6f' % (best_test_loss,best_test_smape))

        best_train_loss_list.append(round(best_train_loss,6))
        best_valid_loss_list.append(round(best_valid_loss,6))
        best_train_smape_list.append(round(best_train_smape,6))
        best_valid_smape_list.append(round(best_valid_smape,6))
        best_test_loss_list.append(round(best_test_loss.item(),6))
        best_test_smape_list.append(round(best_test_smape.item(),6))

    combination = [(v[0].name,v[1].name) for v in product(loss_fun_list,optim_list)]
    recoder['combination'] = combination
    recoder['train_loss'] = best_train_loss_list
    recoder['train_smape'] = best_valid_loss_list
    recoder['valid_loss'] = best_train_smape_list
    recoder['valid_smape'] = best_valid_smape_list
    recoder['test_loss'] = best_test_loss_list
    recoder['test_smape'] = best_test_smape_list
    recoder.to_csv('./output/result.csv')
        # In[ ]:




