#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
import numpy as np
from .base import _BaseAggregator
from .base import _BaseAggregator2
from typing import List, Union
from .base import  BladesClient

def FedAvg(w):
    # w_avg = copy.deepcopy(w[0])
    # for k in w_avg.keys():
    #     for i in range(1, len(w)):
    #         w_avg[k] += w[i][k]
    #     w_avg[k] = torch.div(w_avg[k], len(w))
    w_avg= torch.mean(w, dim=0)
    return w_avg

def Median(w):
    w_avg = torch.median(w, dim=0)[0]
    return w_avg

# def model2vector(model):
#     nparr = np.array([])
#     vec = []
#     for key, var in model.items():
#         if key.split('.')[-1] == 'num_batches_tracked' or key.split('.')[-1] == 'running_mean' or key.split('.')[-1] == 'running_var':
#             continue
#         nplist = var.cpu().numpy()
#         nplist = nplist.ravel()
#         nparr = np.append(nparr, nplist)
#     return nparr

def cos(a,b):
    res=torch.cosine_similarity(a, b, dim=0)
    # res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    '''relu'''
    if res < 0:
        res = 0
    return res

def norm_clip(nparr1, nparr2):
    vnum = torch.linalg.norm(nparr1, ord=None, axis=None, keepdims=False) + 1e-9
    return vnum / torch.linalg.norm(nparr2, ord=None, axis=None, keepdims=False) + 1e-9


def trimmed_mean(all_updates, n_user):
    n_attackers=n_user//5
    sorted_updates = torch.sort(all_updates, 0)[0]
    out = torch.mean(sorted_updates[n_attackers:-n_attackers], 0)
    return out

def multi_krum(all_updates, n_user, multi_k=False):
    n_attackers = n_user // 5
    candidates = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(remaining_updates) > 2 * n_attackers + 2:
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)

        distances = torch.sort(distances, dim=1)[0]
        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]

        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        candidates = remaining_updates[indices[0]][None, :] if not len(candidates) else torch.cat((candidates, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)
        if not multi_k:
            break
    # print(len(remaining_updates))
    aggregate = torch.mean(candidates, dim=0)

    return aggregate


def bulyan(all_updates, nusers):
    print('bulyan used')
    n_attackers = nusers // 5
    bulyan_cluster = []
    candidate_indices = []
    remaining_updates = all_updates
    all_indices = np.arange(len(all_updates))

    while len(bulyan_cluster) < (nusers - 2 * n_attackers):
        torch.cuda.empty_cache()
        distances = []
        for update in remaining_updates:
            distance = []
            for update_ in remaining_updates:
                distance.append(torch.norm((update - update_)) ** 2)
            distance = torch.Tensor(distance).float()
            distances = distance[None, :] if not len(distances) else torch.cat((distances, distance[None, :]), 0)
        # print(distances)

        distances = torch.sort(distances, dim=1)[0]

        scores = torch.sum(distances[:, :len(remaining_updates) - 2 - n_attackers], dim=1)
        indices = torch.argsort(scores)[:len(remaining_updates) - 2 - n_attackers]
        if not len(indices):
            break
        candidate_indices.append(all_indices[indices[0].cpu().numpy()])
        all_indices = np.delete(all_indices, indices[0].cpu().numpy())
        bulyan_cluster = remaining_updates[indices[0]][None, :] if not len(bulyan_cluster) else torch.cat((bulyan_cluster, remaining_updates[indices[0]][None, :]), 0)
        remaining_updates = torch.cat((remaining_updates[:indices[0]], remaining_updates[indices[0] + 1:]), 0)

    # print('dim of bulyan cluster ', bulyan_cluster.shape)

    n, d = bulyan_cluster.shape
    param_med = torch.median(bulyan_cluster, dim=0)[0]
    sort_idx = torch.argsort(torch.abs(bulyan_cluster - param_med), dim=0)
    sorted_params = bulyan_cluster[sort_idx, torch.arange(d)[None, :]]

    return torch.mean(sorted_params[:n - 2 * n_attackers], dim=0)


def norm_bounding(w, nusers):
    #w 50*2472266
    per_norm=torch.norm(w, p=2,dim=1)
    avg_norm=torch.mean(per_norm,dim=0)
    n=per_norm/avg_norm
    mat=torch.max(torch.ones(size=n.size()).cuda(),n)
    mat=mat.reshape(-1,1)
    ans=torch.reciprocal(mat)*w
    return torch.mean(ans, dim=0)

def FLtrust(w, nusers,FLTrustCentralNorm):
    print('FLtrust used')
    FLTrustTotalScore = 0
    sum_parameters = torch.zeros(size=w[0].size()).cuda()
    for client in range(nusers):
        client_score, client_clipped_value = cos(FLTrustCentralNorm, w[client]), norm_clip(FLTrustCentralNorm,w[client])
        FLTrustTotalScore += client_score
        sum_parameters+=client_score * client_clipped_value * w[client]
        # print(str(client) +'...'+ str(client_score) +'...'+ str(client_clipped_value))
    global_parameters = (sum_parameters / FLTrustTotalScore + 1e-9)
    return global_parameters



class Clipping(_BaseAggregator):
    def __init__(self, tau, n_iter=1):
        self.tau = tau
        self.n_iter = n_iter
        super(Clipping, self).__init__()
        self.momentum = None

    def clip(self, v):
        v_norm = torch.norm(v)
        scale = min(1, self.tau / v_norm)
        return v * scale

    def __call__(self, inputs):
        if self.momentum is None:
            self.momentum = torch.zeros_like(inputs[0])

        for _ in range(self.n_iter):
            self.momentum = (
                sum(self.clip(v - self.momentum) for v in inputs) / len(inputs)
                + self.momentum
            )

        # print(self.momentum[:5])
        # raise NotImplementedError
        return torch.clone(self.momentum).detach()

    def __str__(self):
        return "Clipping (tau={}, n_iter={})".format(self.tau, self.n_iter)

def CC(w,net_glob,aggregator):
    fed_model_para = []
    for param in net_glob.parameters():
        fed_model_para = param.data.view(-1) if not len(fed_model_para) else torch.cat(
            (fed_model_para, param.data.view(-1)))
    update=w-fed_model_para

    return fed_model_para+aggregator(update)



class Dnc(_BaseAggregator2):

    def __init__(
        self, num_byzantine=10, *, sub_dim=10000, num_iters=1, filter_frac=1.0
    ) -> None:
        super(Dnc, self).__init__()

        self.num_byzantine = num_byzantine
        self.sub_dim = sub_dim
        self.num_iters = num_iters
        self.fliter_frac = filter_frac

    def __call__(
        self, inputs: Union[List[BladesClient], List[torch.Tensor], torch.Tensor]
    ):
        updates = self._get_updates(inputs)
        d = len(updates[0])

        benign_ids = []
        for i in range(self.num_iters):
            indices = torch.randperm(d)[: self.sub_dim]
            sub_updates = updates[:, indices]
            mu = sub_updates.mean(dim=0)
            centered_update = sub_updates - mu
            v = torch.linalg.svd(centered_update, full_matrices=False)[2][0, :]
            s = np.array(
                [(torch.dot(update - mu, v) ** 2).item() for update in sub_updates]
            )

            good = s.argsort()[
                : len(updates) - int(self.fliter_frac * self.num_byzantine)
            ]
            benign_ids.extend(good)

        benign_ids = list(set(benign_ids))
        benign_updates = updates[benign_ids, :].mean(dim=0)
        return benign_updates


def DNC(w,net_glob):
    fed_model_para = []
    aggregator=Dnc()
    for param in net_glob.parameters():
        fed_model_para = param.data.view(-1) if not len(fed_model_para) else torch.cat(
            (fed_model_para, param.data.view(-1)))
    update=w-fed_model_para

    return fed_model_para+aggregator(update)