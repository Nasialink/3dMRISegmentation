import os
from datetime import datetime
from os.path import join
import argparse
import sys

from torch import sigmoid, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch
import torch.optim as optim

from dataset import MRIDataset_
from loss import DiceLoss
from model import UNet3D
from utils import (get_weight_vector, Report,
                   transfer_weights)

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random



resume_model = False

exp = "./exp_zoom_out" if resume_model else "./exp_zoom_in"
exp_inf = exp + "/inf"
exp_metrics = exp + "/metrics"
exp_figures = exp + "/figures"


config = {
    "sets": "./sets/zoom_in",
    "experiment_root": exp,
    "exp_inf": exp_inf,
    "exp_model_weights": exp,
    "batch_size": 2,
    "num_of_patients": 150,
    "model": exp + "/best_model.pth"
}

if resume_model:
    config["sets"] = "./sets/zoom_out"


is_cuda = torch.cuda.is_available()


net = UNet3D(1, 1, use_bias=True, inplanes=16)

bce_crit = nn.BCELoss()
dice_crit = DiceLoss()
last_bce_loss = 0
last_dice_loss = 0


def criterion(pred, labels, weights=[0.1, 0.9]):
    _bce_loss = bce_crit(pred, labels)
    _dice_loss = dice_crit(pred, labels)
    global last_bce_loss, last_dice_loss
    last_bce_loss = _bce_loss.item()
    last_dice_loss = _dice_loss.item()
    return weights[0] * _bce_loss + weights[1] * _dice_loss


if resume_model:
    size = [144, 172, 71]
else:
    size = [128, 128, 71]

print(size)
assert len(size) == 3
save_dir = config["exp_model_weights"]

w = config["num_of_patients"]
w_tr = int(0.7 * w)
w_vl = int(0.2 * w)
w_ts = int(0.1 * w)
train_x = np.load(config["sets"] + '/train_x.npy')
train_y = np.load(config["sets"] + '/train_y.npy')
train_x = train_x[:w_tr, :, :, :]
train_y = train_y[:w_tr, :, :, :]

val_x = np.load(config["sets"] + '/val_x.npy')
val_y = np.load(config["sets"] + '/val_y.npy')
val_x = val_x[:w_vl, :, :, :]
val_y = val_y[:w_vl, :, :, :]

test_x = np.load(config["sets"] + '/test_x.npy')
test_y = np.load(config["sets"] + '/test_y.npy')
test_x = test_x[:w_ts, :, :, :]
test_y = test_y[:w_ts, :, :, :]

print(train_x.shape, train_y.shape)
print(val_x.shape, val_y.shape)
print(test_x.shape, test_y.shape)

train_loader = DataLoader(MRIDataset_(train_x, train_y), shuffle=True, batch_size=config["batch_size"], pin_memory=True)
val_loader = DataLoader(MRIDataset_(val_x, val_y), shuffle=True, batch_size=config["batch_size"], pin_memory=True)
test_loader = DataLoader(MRIDataset_(val_x, val_y), shuffle=True, batch_size=config["batch_size"], pin_memory=True)








if is_cuda:
    net = net.cuda()
    bce_crit = bce_crit.cuda()
    dice_crit = dice_crit.cuda()




def show_patient(x, y, pred, id):
    plt.figure(figsize=(15, 15)) 
    # l = ['T1', 'T1ce', 'T2', 'Flair', 'Seg']
    s = random.randint(30, 65)
    # for i in range(1, 5):
    plt.subplot(1, 3, 1)
    plt.imshow(x[:, :, s], cmap='gray')
    plt.axis('off')
    plt.title("T1", fontsize=16)
 
    # for i in range(5, 9):
    plt.subplot(1, 3, 2)
    plt.imshow(x[:, :, s], cmap='gray')
    plt.imshow(y[:, :, s], alpha=0.3)
    plt.axis('off')
    plt.title("Target - T1", fontsize=16)

    
    plt.subplot(1, 3, 3)
    plt.imshow(x[:, :, s], cmap='gray')
    plt.imshow(pred[:, :, s], alpha=0.3)
    plt.axis('off')
    plt.title("Pred - T1", fontsize=16)
        
    plt.savefig(config["exp_inf"] + '/p_'+str(id)+'.png')
    plt.close()

def inf(val_loader, epoch, model):
    model.train(False)
    reporter = Report()
    epoch_bce_loss = 0
    epoch_dice_loss = 0
    epoch_loss = 0
    s = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            preds = sigmoid(model(inputs.detach()).detach())
            reporter.feed(preds, labels)

            xs = inputs.cpu().detach().numpy()
            preds = preds.cpu().detach().numpy()
            targets = labels.cpu().detach().numpy()

            # print(xs.shape, preds.shape, targets.shape)
            show_patient(xs[0, 0, :, :, :], targets[0, 0, :, :, :], preds[0, 0, :, :, :], s)
            s += 1
            
    avg_bce_loss = epoch_bce_loss / float(len(val_loader))
    avg_dice_loss = epoch_dice_loss / float(len(val_loader))
    avg_loss = epoch_loss / float(len(val_loader))
    avg_acc = reporter.accuracy()
    print("[Valid] Epoch({}) Avg BCE Loss: {:.4f} Avg Dice Loss: {:.4f} \
        Avg Loss: {:.4f}".format(epoch, avg_bce_loss, avg_dice_loss, avg_loss))
    print(reporter)
    print(reporter.stats())
    return avg_loss, avg_acc

if __name__ == "__main__":
    best_performance = float('Inf')
    epoch = 1 
    # model = torch.load('./models_1200/net-epoch-618.pth')
    model = torch.load(config["model"])
    model.eval()
    
    test_loss, test_acc = inf(test_loader, epoch, model)
    
