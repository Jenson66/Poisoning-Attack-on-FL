from __future__ import print_function
import argparse, os, sys, csv, shutil, time, random, operator, pickle, ast, math
import numpy as np
import pandas as pd
from torch.optim import Optimizer
import torch.nn.functional as F
import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.multiprocessing as mp


def tr_mean(all_updates, n_attackers):
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0) if n_attackers else torch.mean(sorted_updates,0)
    return out

sys.path.insert(0,'./../utils/')
from logger import *
from eval import *
from misc import *

from cifar10_normal_train import *
from cifar10_util import *
from adam import Adam
from sgd import SGD

import torchvision.transforms as transforms
import torchvision.datasets as datasets
data_loc='/mnt/nfs/work1/amir/vshejwalkar/cifar10_data/'
# load the train dataset

train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

cifar10_train = datasets.CIFAR10(root=data_loc, train=True, download=True, transform=train_transform)

cifar10_test = datasets.CIFAR10(root=data_loc, train=False, download=True, transform=train_transform)

X=[]
Y=[]
for i in range(len(cifar10_train)):
    X.append(cifar10_train[i][0].numpy())
    Y.append(cifar10_train[i][1])

for i in range(len(cifar10_test)):
    X.append(cifar10_test[i][0].numpy())
    Y.append(cifar10_test[i][1])

X=np.array(X)
Y=np.array(Y)

print('total data len: ',len(X))

if not os.path.isfile('./cifar10_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices,open('./cifar10_shuffle.pkl','wb'))
else:
    all_indices=pickle.load(open('./cifar10_shuffle.pkl','rb'))

X=X[all_indices]
Y=Y[all_indices]

# data loading

nusers = 50
user_tr_len = 1000

total_tr_len = user_tr_len * nusers
val_len = 5000
te_len = 5000

print('total data len: ', len(X))

if not os.path.isfile('./cifar10_shuffle.pkl'):
    all_indices = np.arange(len(X))
    np.random.shuffle(all_indices)
    pickle.dump(all_indices, open('./cifar10_shuffle.pkl', 'wb'))
else:
    all_indices = pickle.load(open('./cifar10_shuffle.pkl', 'rb'))

total_tr_data = X[:total_tr_len]
total_tr_label = Y[:total_tr_len]

val_data = X[total_tr_len:(total_tr_len + val_len)]
val_label = Y[total_tr_len:(total_tr_len + val_len)]

te_data = X[(total_tr_len + val_len):(total_tr_len + val_len + te_len)]
te_label = Y[(total_tr_len + val_len):(total_tr_len + val_len + te_len)]

total_tr_data_tensor = torch.from_numpy(total_tr_data).type(torch.FloatTensor)
total_tr_label_tensor = torch.from_numpy(total_tr_label).type(torch.LongTensor)

val_data_tensor = torch.from_numpy(val_data).type(torch.FloatTensor)
val_label_tensor = torch.from_numpy(val_label).type(torch.LongTensor)

te_data_tensor = torch.from_numpy(te_data).type(torch.FloatTensor)
te_label_tensor = torch.from_numpy(te_label).type(torch.LongTensor)

print('total tr len %d | val len %d | test len %d' % (
len(total_tr_data_tensor), len(val_data_tensor), len(te_data_tensor)))

# ==============================================================================================================

user_tr_data_tensors = []
user_tr_label_tensors = []

for i in range(nusers):
    user_tr_data_tensor = torch.from_numpy(total_tr_data[user_tr_len * i:user_tr_len * (i + 1)]).type(torch.FloatTensor)
    user_tr_label_tensor = torch.from_numpy(total_tr_label[user_tr_len * i:user_tr_len * (i + 1)]).type(
        torch.LongTensor)

    user_tr_data_tensors.append(user_tr_data_tensor)
    user_tr_label_tensors.append(user_tr_label_tensor)
    print('user %d tr len %d' % (i, len(user_tr_data_tensor)))


batch_size=250
resume=0
nepochs=1200
schedule=[1000]
nbatches = user_tr_len//batch_size

gamma=.5
opt = 'sgd'
fed_lr=0.5
criterion=nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()

aggregation='trmean'
multi_k = False
candidates = []

at_type='LIE'
z_values={3:0.69847, 5:0.7054, 8:0.71904, 10:0.72575, 12:0.73891}
n_attackers=[0]

arch='resnet'
chkpt='./'+aggregation

for n_attacker in n_attackers:
    epoch_num = 0
    best_global_acc = 0
    best_global_te_acc = 0
    torch.cuda.empty_cache()
    r=np.arange(user_tr_len)

    fed_model, _ = return_model(arch, 0.1, 0.9, parallel=False)
    optimizer_fed = SGD(fed_model.parameters(), lr=fed_lr)

    while epoch_num <= nepochs:
        user_grads=[]
        if not epoch_num and epoch_num%nbatches == 0:
            # np.random.shuffle(r)

            for i in range(nusers):
                user_tr_data_tensors[i]=user_tr_data_tensors[i][r]
                user_tr_label_tensors[i]=user_tr_label_tensors[i][r]

        for i in range(n_attacker, nusers):

            inputs = user_tr_data_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
            targets = user_tr_label_tensors[i][(epoch_num%nbatches)*batch_size:((epoch_num%nbatches) + 1) * batch_size]
            # print(inputs.shape)
            b = targets >=2
            targets=targets[b]
            inputs=inputs[b]

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            outputs = fed_model(inputs)
            loss = criterion(outputs, targets)
            fed_model.zero_grad()
            loss.backward(retain_graph=True)

            param_grad=[]
            for param in fed_model.parameters():
                param_grad=param.grad.data.view(-1) if not len(param_grad) else torch.cat((param_grad,param.grad.view(-1)))

            user_grads=param_grad[None, :] if len(user_grads)==0 else torch.cat((user_grads,param_grad[None,:]), 0)

        malicious_grads = user_grads

        if epoch_num in schedule:
            for param_group in optimizer_fed.param_groups:
                param_group['lr'] *= gamma
                print('New learnin rate ', param_group['lr'])

        if n_attacker > 0:
            if at_type == 'lie':
                mal_update = lie_attack(malicious_grads, z_values[n_attacker])
                malicious_grads = torch.cat((torch.stack([mal_update]*n_attacker), malicious_grads))
            elif at_type == 'fang':
                agg_grads = torch.mean(malicious_grads, 0)
                deviation = torch.sign(agg_grads)
                malicious_grads = get_malicious_updates_fang_trmean(malicious_grads, deviation, n_attacker, epoch_num)
            elif at_type == 'our-agr':
                agg_grads = torch.mean(malicious_grads, 0)
                malicious_grads = our_attack_krum(malicious_grads, agg_grads, n_attacker, compression=compression, q_level=q_level, norm=norm)

        if not epoch_num :
            print(malicious_grads.shape)

        if aggregation=='median':
            agg_grads=torch.median(malicious_grads,dim=0)[0]

        elif aggregation=='average':
            agg_grads=torch.mean(malicious_grads,dim=0)

        elif aggregation=='trmean':
            agg_grads=tr_mean(malicious_grads, 10)

        elif aggregation=='krum' or aggregation=='mkrum':
            multi_k = True if aggregation == 'mkrum' else False
            if epoch_num == 0: print('multi krum is ', multi_k)
            agg_grads, krum_candidate = multi_krum(malicious_grads, n_attacker, multi_k=multi_k)

        elif aggregation=='bulyan':
            agg_grads, krum_candidate=bulyan(malicious_grads, n_attacker)

        del user_grads

        start_idx=0

        optimizer_fed.zero_grad()

        model_grads=[]

        for i, param in enumerate(fed_model.parameters()):
            param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx=start_idx+len(param.data.view(-1))
            param_=param_.cuda()
            model_grads.append(param_)

        optimizer_fed.step(model_grads)

        val_loss, val_acc = test(val_data_tensor,val_label_tensor,fed_model,criterion,use_cuda)
        te_loss, te_acc = test(te_data_tensor,te_label_tensor, fed_model, criterion, use_cuda)

        is_best = best_global_acc < val_acc

        best_global_acc = max(best_global_acc, val_acc)

        if is_best:
            best_global_te_acc = te_acc

        if epoch_num%10==0 or epoch_num==nepochs-1:
            print('%s: at %s n_at %d e %d fed_model val loss %.4f val acc %.4f best val_acc %f te_acc %f'%(aggregation, at_type, n_attacker, epoch_num, val_loss, val_acc, best_global_acc,best_global_te_acc))

        if val_loss > 10:
            print('val loss %f too high'%val_loss)
            break

        epoch_num+=1
