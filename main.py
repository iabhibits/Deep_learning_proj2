import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

from Fashion_mnist_cnn import Model as conv_Model
from Fashion_mnist_linear import Model as mlp_Model
from Fashion_mnist_cnn import preprocess_data

def simple_accuracy(preds, labels):
    #preds = np.argmax(preds, axis = 0)
    r = []
    for i in range(preds.shape[0]):
        r.append(np.argmax(preds[i]))
    #print(r)
    #print("#######################")
    #print(labels)
    return np.mean(np.array(r) == np.array(labels)),r

def evaluate(model,eval_dataset,device, test = False):
    results = {}
    preds = None
    labels = None
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size = 32)
    eval_loss = 0
    nb_eval_steps = 0
    best_steps = 0
    best_eval_acc = 0
    for batch in eval_dataloader:

        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = { 
                        "feature":batch[0].unsqueeze(dim=1),
                        "labels": batch[1],
                    }
            outputs = model(**inputs)
            #print(outputs)
            tmp_eval_loss = outputs[0]
            logits = outputs[1]
            #print(logits)
            #print("###########################")
            #arr = np.array(logits)
            #print(logits.shape)
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                #print("preds.hsape",preds.shape)
                labels = inputs["labels"].detach().cpu().numpy()
                #print("\n")
                #print(labels.shape)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                #print("preds.shape",preds.shape)
                #print("##########################################################")
                labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)
                #print("labels shape",labels.shape)
        eval_loss = eval_loss / nb_eval_steps
        
    acc, r = simple_accuracy(preds, labels)
    result = {"acc": acc, "eval_loss": eval_loss}
    results.update(result)
    print(result)
    #print("##############",device)
    
    '''if device is 'cpu':
        images,labels = batch[0],r
        s = 0
        l = 16
        for i in range(eval_batch_size//16):
            visualize_multi(images,labels,16,s)
            s = l -1
            l = l + eval_batch_size // 2'''
    if test == True:
        return eval_loss, results, r, labels


    return eval_loss, results

def linear_evaluate(model,eval_dataset,device,test=False):
    results = {}
    preds = None
    labels = None
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size = 32)
    eval_loss = 0
    nb_eval_steps = 0
    best_steps = 0
    best_eval_acc = 0
    for batch in tqdm(eval_dataloader):

        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = { 
                        "feature":batch[0].view(-1,784),
                        "labels": batch[1],
                    }
            outputs = model(**inputs)
            #print(outputs)
            tmp_eval_loss = outputs[0]
            logits = outputs[1]
            #print(logits)
            #print("###########################")
            #arr = np.array(logits)
            #print(logits.shape)
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                #print("preds.hsape",preds.shape)
                labels = inputs["labels"].detach().cpu().numpy()
                #print("\n")
                #print(labels.shape)
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                #print("preds.shape",preds.shape)
                #print("##########################################################")
                labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)
                #print("labels shape",labels.shape)
        eval_loss = eval_loss / nb_eval_steps
        
    acc, r = simple_accuracy(preds, labels)
    result = {"acc": acc, "eval_loss": eval_loss}
    results.update(result)
    print(result)
    #print("##############",device)
    
    '''if device is 'cpu':
        images,labels = batch[0],r
        s = 0
        l = 16
        for i in range(eval_batch_size//16):
            visualize_multi(images,labels,16,s)
            s = l -1
            l = l + eval_batch_size // 2'''
    if test == True:
        return eval_loss, results, r, labels
    return results

def main():
    print("Abhishek Kumar")
    print("15648")
    print("CSA")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--test-data", default='test_input.txt', type=str, help="Name of the test file")
    # args = parser.parse_args()
    
    mlp_model_path='model/MLP.pt'
    conv_model_path = 'model/convNet.pt'
    mlp_model = mlp_Model()
    conv_model = conv_Model()
    device = "cpu"
    mlp_model.to(device)
    conv_model.to(device)
    train_dataset, test_dataset, eval_dataset = preprocess_data()
    mlp_model.load_state_dict(torch.load("./models/MLP.pt"))
    conv_model.load_state_dict(torch.load("./models/convNet.pt"))
    loss, results,pred ,target = linear_evaluate(mlp_model,test_dataset,device,True)
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(results['acc']))
        f.write("gt_label,pred_label \n")
        for idx in range(len(target)):
            f.write("{},{}\n".format(target[idx], pred[idx]))

    #print(cmodel)
    loss, results,pred ,target =evaluate(conv_model,test_dataset,device,True)

    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(results['acc']))
        f.write("gt_label,pred_label \n")
        for idx in range(len(target)):
            f.write("{},{}\n".format(target[idx], pred[idx]))
if __name__ == '__main__':
    main()