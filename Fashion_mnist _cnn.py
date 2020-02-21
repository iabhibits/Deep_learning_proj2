#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[7]:


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


# In[8]:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# In[9]:


def visualize_multi(images,labels,n,s):
    fig = plt.figure()
    names = ["top", "trouser", "pullover", "dress", "coat",
                  "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    for index in range(1,17):
        ax = plt.subplot(4,4,index)
        plt.axis('off')
        index = s+index
        ax.imshow(images[index].numpy().squeeze(),cmap = 'Greys_r')
        ax.set_title(str(names[labels[index].squeeze()]))


# In[10]:


def visualize_images(data):
    if data.shape[0] == 1:
        plt.imshow(data.squeeze(),cmap = 'Greys_r')
    else:
        plt.imshow(x,cmap = 'Greys_r')


# In[11]:


def convert_data(data):
    data = data.values
    label = data[:,:0]
    feature = data[:,1:]
    feature = [f.reshape(1,28,28) for f in feature]
    return label, feature


# In[12]:


def preprocess_data():
    transform = transforms.Compose([transforms.ToTensor(),])
    train_data = datasets.FashionMNIST("./dataset/", train=True,transform=transform, download=True)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42,shuffle=True)
    test_data = datasets.FashionMNIST("./dataset/",train=False, transform=transform, download=True)
    return train_data, test_data, val_data


# In[14]:


class Model(nn.Module):
    def __init__(self):
        emb_dim = 20
        n_classes = 10
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64,128,kernel_size=7,stride=1,padding=3)
        self.emb = nn.Linear(128*7*7, 100)
        self.drop = nn.Dropout(0.25)
        self.clf = nn.Linear(100, n_classes)

    def forward(self,feature=None, labels=None):
        x = feature.view(-1, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128*7*7)
        x = self.drop(x)
        x = self.emb(x)
        logits = self.clf(x)
        outputs = (logits,)
        
        if labels is None:
            loss = 0
            outputs = (loss, ) + outputs

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (loss, ) + outputs
        return outputs


# In[15]:


def simple_accuracy(preds, labels):
    #preds = np.argmax(preds, axis = 0)
    r = []
    for i in range(preds.shape[0]):
        r.append(np.argmax(preds[i]))
    #print(r)
    #print("#######################")
    #print(labels)
    return np.mean(np.array(r) == np.array(labels)), r


# In[16]:


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
                        "feature":batch[0],
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
            if step % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                t, step * len(batch[0]), len(train_dataloader.dataset),
                100. * step / len(train_dataloader), loss.item()))
        
            results = evaluate(model,eval_dataset,device)
            if results['acc'] > best_eval_acc:
                best_eval_acc = results['acc']
                best_steps = global_step
            #for key, value in results.items():
                #tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        
    #tb_writer.close()

    print("best_eval_acc.{}\n".format(best_eval_acc))
    print("Best steps {}\n".format(best_steps))
    return global_step, tr_loss / global_step, best_steps


# In[17]:


eval_batch_size = 32


# In[18]:


def evaluate(model,eval_dataset,device, test = False):
    results = {}
    preds = None
    labels = None
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size = eval_batch_size)
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
    #print(result)
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


# In[19]:


train_batch_size = 2048
num_train_epochs = 300
save_steps = 2000
output_dir = './model/'
do_train = True
do_eval =True


# In[20]:


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


# In[21]:


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
        #     output_dir = os.path.join(output_dir, "model.bin")
        # else :
        #     output_dir = os.path.join(output_dir, "model.bin")
        # torch.save(model.state_dict(), output_dir)
        evaluate(model, test_dataset,device)
    return model


# In[ ]:


if __name__ == "__main__":
    model = main()
    output_dir = os.path.join(output_dir, "model.bin")
    torch.save(model, output_dir)


# In[ ]:




