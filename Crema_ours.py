#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import defaultdict
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
import os
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
import json
import numpy as np
import argparse
import random
from sklearn.metrics import f1_score, average_precision_score
from data.template import config
from dataset.CREMA import CramedDataset
from model.AudioVideo import AVClassifier
from utils.stocBiO import *
from utils.min_norm_solvers import MinNormSolver
from utils.utils import (
    create_logger,
    Averager,
    deep_update_dict,
)
from utils.tools import weight_init

def compute_mAP(outputs, labels):
    y_true = labels.cpu().detach().numpy()
    y_pred = outputs.cpu().detach().numpy()
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(AP)


def Alignment(a_f, v_f):
    a_f = F.normalize(a_f, p=2, dim=-1)
    v_f = F.normalize(v_f, p=2, dim=-1)
    similarity_matrix = torch.matmul(a_f, v_f.T) 
        
    similarity_matrix = similarity_matrix / 0.07
    
    labels = torch.arange(a_f.size(0)).to(a_f.device)
    loss_a_to_v = F.cross_entropy(similarity_matrix, labels)
    loss_v_to_a = F.cross_entropy(similarity_matrix.T, labels)
    loss = (loss_a_to_v + loss_v_to_a) / 2.0
    return loss

def Alignment(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    similarity_matrix_qp = torch.matmul(q, p.T)
    similarity_matrix_pq = torch.matmul(p, q.T)
    labels = torch.arange(p.size(0)).long().cuda()
    loss_qp = F.cross_entropy(similarity_matrix_qp, labels)
    loss_pq =  F.cross_entropy(similarity_matrix_pq, labels)
    loss = 0.1 * loss_qp + 0.2 * loss_pq
    return loss

def Alignment(p, q):
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * (p + q)
    kl_p_m = F.kl_div(p.log(), m, reduction='batchmean')
    kl_q_m = F.kl_div(q.log(), m, reduction='batchmean')
    js_score = 0.5 * (kl_p_m + kl_q_m)
    return js_score

def getAlpha_grad(loss_alignment, loss_cls):
    loss_cls.backward(retain_graph=True)
    grads_cls = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    optimizer.zero_grad()
    loss_alignment.backward(retain_graph=True)
    grads_alignment = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
    optimizer.zero_grad()
    this_cos = sum(
        (grads_cls[name] * grads_alignment[name]).sum().item()
        for name in grads_cls.keys() & grads_alignment.keys()
    )
    cls_k = [0.5, 0.5]
    if this_cos >= 0: 
        cls_k, _ = MinNormSolver.find_min_norm_element(
            [list(grads_cls.values()), list(grads_alignment.values())]
        )
    return cls_k


def getAlpha_heritic(epoch, a = 0.3):
    alpha = np.exp(-a * epoch)
    alpha = np.clip(alpha, 0.05, 0.95)
    return [alpha,  1.0 -alpha]


def train_audio_video(epoch, train_loader, model, optimizer, logger):
    model.train()
    tl = Averager()
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()


    for step, (spectrogram, image, y) in enumerate(tqdm(train_loader)):
        image = image.float().cuda()
        y = y.cuda()
        spectrogram = spectrogram.unsqueeze(1).float().cuda()

        optimizer.zero_grad()

        o_b, o_a, o_v, a_f, v_f = model(spectrogram, image)

        loss_cls = criterion((o_a * 0.5 + o_v * 0.5), y).mean()
        loss_alignment = Alignment(a_f, v_f) # learning rate 1e-4,
        # loss_alignment = Alignment(o_a, o_v)

        # cls_k = getAlpha_grad(loss_alignment, loss_cls) # learning rate 1e-3,
        cls_k = getAlpha_heritic(epoch)   # learng rate 1e-4,
        loss = cls_k[0] * loss_alignment + cls_k[1] * loss_cls

        loss.backward()
        optimizer.step()

        tl.add(loss.item())


    loss_ave = tl.item()

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: Average Training Loss:{loss_ave:.3f}').format(epoch=epoch, loss_ave=loss_ave))

    return model


def val(epoch, val_loader, model, logger):

    model.eval()
    pred_list = []
    pred_list_a = []
    pred_list_v = []
    label_list = []
    soft_pred = []
    soft_pred_a = []
    soft_pred_v = []
    one_hot_label = []
    score_a = 0.0
    score_v = 0.0
    with torch.no_grad():
        for step, (spectrogram, image, y) in enumerate(tqdm(val_loader)):
            label_list = label_list + torch.argmax(y, dim=1).tolist()
            one_hot_label = one_hot_label + y.tolist()
            image = image.cuda()
            y = y.cuda()
            spectrogram = spectrogram.unsqueeze(1).float().cuda()

            _ ,o_a, o_v,_,_ = model(spectrogram, image)

            soft_pred_a = soft_pred_a + (F.softmax(o_a, dim=1)).tolist()
            soft_pred_v = soft_pred_v + (F.softmax(o_v, dim=1)).tolist()
            soft_pred = soft_pred + (F.softmax((o_a * 0.5 +o_v *0.5), dim=1)).tolist()
            pred = (F.softmax((o_a * 0.5 +o_v *0.5), dim=1)).argmax(dim=1)
            pred_a = (F.softmax(o_a, dim=1)).argmax(dim=1)
            pred_v = (F.softmax(o_v, dim=1)).argmax(dim=1)

            pred_list = pred_list + pred.tolist()
            pred_list_a = pred_list_a + pred_a.tolist()
            pred_list_v = pred_list_v + pred_v.tolist()

        f1 = f1_score(label_list, pred_list, average='macro')
        f1_a = f1_score(label_list, pred_list_a, average='macro')
        f1_v = f1_score(label_list, pred_list_v, average='macro')
        correct = sum(1 for x, y in zip(label_list, pred_list) if x == y)
        correct_a = sum(1 for x, y in zip(label_list, pred_list_a) if x == y)
        correct_v = sum(1 for x, y in zip(label_list, pred_list_v) if x == y)
        acc = correct / len(label_list)
        acc_a = correct_a / len(label_list)
        acc_v = correct_v / len(label_list)
        mAP = compute_mAP(torch.Tensor(soft_pred), torch.Tensor(one_hot_label))
        mAP_a = compute_mAP(torch.Tensor(soft_pred_a), torch.Tensor(one_hot_label))
        mAP_v = compute_mAP(torch.Tensor(soft_pred_v), torch.Tensor(one_hot_label))

    logger.info('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    logger.info(('Epoch {epoch:d}: f1:{f1:.4f},acc:{acc:.4f},mAP:{mAP:.4f},f1_a:{f1_a:.4f},acc_a:{acc_a:.4f},mAP_a:{mAP_a:.4f},f1_v:{f1_v:.4f},acc_v:{acc_v:.4f},mAP_v:{mAP_v:.4f}').format(epoch=epoch, f1=f1, acc=acc, mAP=mAP,
                                                                                                                                                                                            f1_a=f1_a, acc_a=acc_a, mAP_a=mAP_a,
                                                                                                                                                                                            f1_v=f1_v, acc_v=acc_v, mAP_v=mAP_v))
    return acc, score_a, score_v

if __name__ == '__main__':
    # ----- LOAD PARAM -----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default='./data/crema.json')

    args = parser.parse_args()
    cfg = config

    with open(args.config, "r") as f:
        exp_params = json.load(f)

    cfg = deep_update_dict(exp_params, cfg)

    # ----- SET SEED -----
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu_id']
    # ----- SET LOGGER -----
    local_rank = cfg['train']['local_rank']
    logger, log_file, exp_id = create_logger(cfg, local_rank)

    # ----- SET DATALOADER -----
    train_dataset = CramedDataset(config, mode='train')
    test_dataset = CramedDataset(config, mode='test')

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg['train']['batch_size'], shuffle=True,
                              num_workers=cfg['train']['num_workers'], pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=cfg['test']['batch_size'], shuffle=False,
                             num_workers=cfg['test']['num_workers'], pin_memory=True)

                            
    # ----- MODEL -----
    model = AVClassifier(config=cfg)
    model = model.cuda()
    model.load_state_dict(torch.load('./pretrain/pretrain.pth'))


    lr_adjust = config['train']['optimizer']['lr']

    optimizer = optim.SGD(model.parameters(), lr=lr_adjust,
                          momentum=config['train']['optimizer']['momentum'],
                          weight_decay=config['train']['optimizer']['wc'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, config['train']['lr_scheduler']['patience'], 0.1)
    best_acc = 0


    for epoch in range(20):
        logger.info(('Epoch {epoch:d} is pending...').format(epoch=epoch))

        scheduler.step()
        model = train_audio_video(epoch, train_loader, model, optimizer, logger)

        acc, v_a, v_v = val(epoch, test_loader, model, logger)

        if acc > best_acc:
            best_acc = acc
            print('Find a better model and save it!')
            logger.info('Find a better model and save it!')
            torch.save(model.state_dict(), 'Crema_best_model.pth')