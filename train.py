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

resume_model = True

exp = "./exp_zoom_out" if resume_model else "./exp_zoom_in"
exp_inf = exp + "/inf"
exp_metrics = exp + "/metrics"
exp_figures = exp + "/figures"

if not resume_model:
    config = {
        "sets": "./sets/zoom_in",
        "experiment_root": exp,
        "exp_inf": exp_inf,
        "exp_metrics": exp_metrics,
        "exp_figures": exp_figures,
        "exp_model_weights": exp,
        "batch_size": 2,
        "epochs": 1200,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "weight": 1,
        "restart": 50,
        "num_of_patients": 150
    }
else: 
    config = {
        "sets": "./sets/zoom_out",
        "experiment_root": exp,
        "exp_inf": exp_inf,
        "exp_metrics": exp_metrics,
        "exp_figures": exp_figures,
        "exp_model_weights": exp,
        "batch_size": 2,
        "epochs": 150,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "weight": 1,
        "restart": 50,
        "num_of_patients": 150
    }

try:  
    os.mkdir(config["experiment_root"])
    os.mkdir(config["exp_inf"])
    os.mkdir(config["exp_metrics"])
    os.mkdir(config["exp_figures"])
    # os.mkdir(config["exp_model_weights"]) 
except OSError as error:  
    print(error)  

is_cuda = torch.cuda.is_available()

model_weights_path = './exp_zoom_in/best_model.pth'

net = UNet3D(1, 1, use_bias=True, inplanes=16)
if resume_model:
    transfer_weights(net, model_weights_path)
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
relative_weight = config["weight"]
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




optimizer = optim.Adam(net.parameters(),
                       lr=config["learning_rate"],
                       weight_decay=config["weight_decay"])
scheduler = CosineAnnealingLR(optimizer,
                              T_max=config["restart"] * len(train_loader))

f_tr = open(config["exp_metrics"] + "/train.txt", "a")
f_vl = open(config["exp_metrics"] + "/val.txt", "a")
f_ts = open(config["exp_metrics"] + "/test.txt", "a")


if is_cuda:
    net = net.cuda()
    bce_crit = bce_crit.cuda()
    dice_crit = dice_crit.cuda()


def train(train_loader, epoch):
    net.train(True)
    reporter = Report()
    epoch_bce_loss = 0
    epoch_dice_loss = 0
    epoch_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        if is_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = sigmoid(net(inputs))
        reporter.feed(outputs, labels)
        bce_crit.weight = get_weight_vector(labels, relative_weight, is_cuda)
        loss = criterion(outputs, labels)
        epoch_bce_loss += last_bce_loss
        epoch_dice_loss += last_dice_loss
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        del inputs, labels, outputs, loss
    avg_bce_loss = epoch_bce_loss / float(len(train_loader))
    avg_dice_loss = epoch_dice_loss / float(len(train_loader))
    avg_loss = epoch_loss / float(len(train_loader))
    avg_acc = reporter.accuracy()
    print("\n[Train] Epoch({}) Avg BCE Loss: {:.4f} Avg Dice Loss: {:.4f} \
        Avg Loss: {:.4f}".format(epoch, avg_bce_loss, avg_dice_loss, avg_loss))
    
    print(reporter)
    print(reporter.HD, reporter.SD)
    f_tr.write(str(reporter.HD)+" "+str(reporter.SD)+"\n")
    # print(reporter.stats())
    return avg_loss, avg_acc


def validate(val_loader, epoch):
    net.train(False)
    reporter = Report()
    epoch_bce_loss = 0
    epoch_dice_loss = 0
    epoch_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            preds = sigmoid(net(inputs.detach()).detach())
            reporter.feed(preds, labels)
            bce_crit.weight = get_weight_vector(labels, relative_weight,
                                                is_cuda)
            loss = criterion(preds, labels)
            epoch_bce_loss += last_bce_loss
            epoch_dice_loss += last_dice_loss
            epoch_loss += loss.item()
            del inputs, labels, preds, loss
    avg_bce_loss = epoch_bce_loss / float(len(val_loader))
    avg_dice_loss = epoch_dice_loss / float(len(val_loader))
    avg_loss = epoch_loss / float(len(val_loader))
    avg_acc = reporter.accuracy()
    print("[Valid] Epoch({}) Avg BCE Loss: {:.4f} Avg Dice Loss: {:.4f} \
        Avg Loss: {:.4f}".format(epoch, avg_bce_loss, avg_dice_loss, avg_loss))
    
    print(reporter)
    print(reporter.HD, reporter.SD)
    f_vl.write(str(reporter.HD)+" "+str(reporter.SD)+"\n")
    # print(reporter.stats())
    return avg_loss, avg_acc


def inf(val_loader, epoch):
    net.train(False)
    reporter = Report()
    epoch_bce_loss = 0
    epoch_dice_loss = 0
    epoch_loss = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()
            preds = sigmoid(net(inputs.detach()).detach())
            reporter.feed(preds, labels)
            bce_crit.weight = get_weight_vector(labels, relative_weight,
                                                is_cuda)
            loss = criterion(preds, labels)
            epoch_bce_loss += last_bce_loss
            epoch_dice_loss += last_dice_loss
            epoch_loss += loss.item()
            del inputs, labels, preds, loss
    avg_bce_loss = epoch_bce_loss / float(len(val_loader))
    avg_dice_loss = epoch_dice_loss / float(len(val_loader))
    avg_loss = epoch_loss / float(len(val_loader))
    avg_acc = reporter.accuracy()
    print("[Valid] Epoch({}) Avg BCE Loss: {:.4f} Avg Dice Loss: {:.4f} \
        Avg Loss: {:.4f}".format(epoch, avg_bce_loss, avg_dice_loss, avg_loss))
    print(reporter)
    print(reporter.stats())
    f_ts.write(str(reporter.HD)+" "+str(reporter.SD)+"\n")
    return avg_loss, avg_acc

if __name__ == "__main__":
    best_performance = float('Inf')
    n_epochs = config["epochs"]
    for epoch in tqdm(range(n_epochs)):
        print("")
        print("")
        print("----------  Training  -----------")
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        train_loss, train_acc = train(train_loader, epoch)
        
        print("")
        print("")
        print("----------  Validation  -----------")
        valid_loss, valid_acc = validate(val_loader, epoch)

        if valid_loss < best_performance:
            best_performance = valid_loss
            torch.save(
                net,
                join(save_dir, 'best_model.pth'.format(epoch)))
            print("model saved")
        if epoch > config["restart"] and epoch % config["restart"] == 0:
            scheduler.last_epoch = -1
            print("lr restart")
        sys.stdout.flush()
    print("")
    print("")
    print("----------  Inference  -----------")
    test_loss, test_acc = inf(test_loader, epoch)

    f_tr.close()
    f_vl.close()
    f_ts.close()
    
