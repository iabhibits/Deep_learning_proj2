#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import glob
import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange

import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
#import model as Model


# In[2]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[47]:


import matplotlib as mlp
import matplotlib.pyplot as plt


# In[85]:


import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[3]:


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


# In[4]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# In[5]:


def visualize_images(data):
    if data.shape[0] == 1:
        plt.imshow(data.squeeze(),cmap = 'Greys_r')
    else:
        plt.imshow(x,cmap = 'Greys_r')


# In[6]:


def visualize_multi(images,labels,n,s):
    fig = plt.figure()
    names = ["top", "trouser", "pullover", "dress", "coat",
                  "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    for index in range(1,10):
        ax = plt.subplot(3,3,index)
        plt.axis('off')
        index = s+index
        ax.imshow(images[index].numpy().squeeze(),cmap = 'Greys_r')
        ax.set_title(str(names[labels[index].squeeze()]))
        fig.savefig("./visualize_fmnist.png")


# In[7]:


def convert_data(data):
    data = data.values
    label = data[:,:0]
    feature = data[:,1:]
    feature = [f.reshape(1,28,28) for f in feature]
    return label, feature


# In[8]:


def preprocess_data():
    transform = transforms.Compose([transforms.ToTensor(),])
    train_data = datasets.FashionMNIST("./dataset/", train=True,transform=transform, download=True)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    test_data = datasets.FashionMNIST("./dataset/", transform=transform, download=True)
    return train_data, test_data, val_data


# In[9]:


def simple_accuracy(preds, labels):
    #preds = np.argmax(preds, axis = 0)
    r = []
    for i in range(preds.shape[0]):
        r.append(np.argmax(preds[i]))
    #print(r)
    #print("#######################")
    #print(labels)
    return np.mean(np.array(r) == np.array(labels)),r


# In[40]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(784, 1024)
        self.drop = nn.Dropout(0.5)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512,128)
        self.classifier = nn.Linear(128,10)

    def forward(self,feature=None, labels=None):
        x = F.relu(self.layer1(feature))
        x = self.drop(x)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        logits = self.classifier(x)
        outputs = (logits,)
        
        if labels is None:
            loss = 0
            outputs = (loss, ) + outputs

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss, ) + outputs
        return outputs


# In[78]:


def train(train_dataset,eval_dataset, model,device):
    #tb_writer = SummaryWriter()
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size = train_batch_size)
    t_total = len(train_dataloader) // num_train_epochs

    log_interval = 1000
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, betas=[0.9,0.98])

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_eval_acc = 0.0
    best_steps = 0
    model.zero_grad()

    train_iterator = trange(int(num_train_epochs), desc="Epoch")
    #set_seed(100)

    for t in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = { 
                        "feature":batch[0].view(-1,784),
                        "labels": batch[1],
            }

            outputs = model(**inputs)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            #tb_writer.add_scalar("Training loss", (loss.item()), global_step)
            model.zero_grad()
            global_step += 1
            '''if step % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                t, step * len(batch[0]), len(train_dataloader.dataset),
                100. * step / len(train_dataloader), loss.item()))'''
        
            results = evaluate(model,eval_dataset,device)
            if results['acc'] > best_eval_acc:
                best_eval_acc = results['acc']
                best_steps = global_step
            #for key, value in results.items():
                #tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        
    #tb_writer.close()
        print("Train epoch {} {}\n".format(t,results))

    print("best_eval_acc.{}\n".format(best_eval_acc))
    print("Best steps {}\n".format(best_steps))
    return global_step, tr_loss / global_step, best_steps


# In[79]:


eval_batch_size = 32


# In[80]:


def evaluate(model,eval_dataset,device,test=False):
    results = {}
    preds = None
    labels = None
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size = eval_batch_size)
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
        return r, labels
    return results


# In[98]:


train_batch_size = 1024
num_train_epochs = 100
output_dir = './model/'
do_train = True
do_eval =True


# In[99]:


def draw_confusionMatrix(target_labels, Predicted_labels):
    classNames = ['T-shirt/top','Trouser','Pullover','Dress','Coat', 'Sandal','Shirt','Sneaker','Bag','Ankle boot']
    mat = confusion_matrix(np.array(target_labels), np.array(Predicted_labels))
    data = pd.DataFrame(mat, range(len(mat)), range(len(mat)))
    fig = plt.figure(figsize=(12,12))
    sns.set(font_scale=1)
    heat_map = sns.heatmap(data, annot=True, annot_kws={"size": 8}, xticklabels = classNames, yticklabels = classNames,cmap='BuPu' )
    bottom, top = heat_map.get_ylim()
    heat_map.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig("Confusion Matrix MLP.png")
    plt.show()


# In[100]:


def main():
    #parser = argparse.ArgumentParser()

    # Required Parameters

    #parser.add_argument("--train_batch_size", default=32, type=int, help="What is the training batch size?")
    #parser.add_argument("--num_train_epochs", default=32, type=int, help="Total number of training epochs to perform.")
    #parser.add_argument("--logging_steps", default=500, type=int, help="Log every X update steps")
    #parser.add_argument("--save_steps", default=500, type=int, help="Save checkpoints every X steps")
    #parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    #parser.add_argument("--output_dir", default = 'save/', type=str, help="The output directory where the model predictions and checkpoints will be written.")
    #parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    #parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")


    #args = parser.parse_args()
    train_dataset, test_dataset, eval_dataset = preprocess_data()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = device
    print("The cuda device being used is : {} \n".format(device))
    seed = 42
    set_seed(seed)

    if do_train:
        model = Model()
        model.to(device)
        global_step, tr_loss, best_steps = train(train_dataset,eval_dataset, model,device)
        logger.info("global_steps = %s, average_loss = %s ", global_step, tr_loss)
        predicted, target = evaluate(model, test_dataset,device,True)
        draw_confusionMatrix(target,predicted)
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)
        #     output_dir = os.path.join(output_dir, "MLP.pt")
        # else :
        #     output_dir = os.path.join(output_dir, "MLP.pt")
        # torch.save(model, output_dir)
    return model


# In[ ]:


if __name__ == "__main__":
    model = main()
    output_dir = os.path.join(output_dir, "MLP.pt")
    torch.save(model, output_dir)

# In[75]:




# In[ ]:




