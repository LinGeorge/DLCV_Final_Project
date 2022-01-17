# built-in packages
import sys
import time
import csv
import os
import random
import shutil
# allowed-by-TAs third-party packages
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
from torchvision.transforms.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from torchvision import models
from sklearn.manifold import TSNE

import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

import os
import random
import shutil
import time
import warnings
import PIL
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torchsampler import ImbalancedDatasetSampler

import timm



# fix random seeds
def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print('Seeds are now fixed.')

fix_seeds(0)


# the configuration dictionary for p1
cfg = {
    # the cfg['DEVICE'] to be used for training and validation
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # the prefix for avoiding overwriting when backuping the checkpoint and output csv-files
    'BACKUP_PREFIX': 'nowd',
    
    # the number of classes
    'N_CLS': 1000,

    # the batch-size
    'BATCH_SZ': 32,
    
    # the number of epochs during training
    'N_EPOCHS': 10,

    # the name of the pretrained model under torchvision to be used
    'PRETRAINED_MODEL_NAME': 'resnet50',
    
    # the name of the optimizer used in training
    'OPTIM_NAME': 'Adam',
    
    # some parameters of the used optimizer
    'OPTIM_PARAMS': {
        'lr': 0.00005,
        # 'weight_decay': 1e-3,
    },
    
    # to determine whether to use lr-scheduler or not
    'USE_SCHED': True,
    
    # the name of the lr-scheduler used in training
    'SCHED_NAME': 'ReduceLROnPlateau',

    # the max-norm used in gradients clipping
    'MAX_NORM': 1,
    
    # some parameters of the used lr-scheduler
    'SCHED_PARAMS': {
        'factor': 0.5,
        'patience': 1,
        'min_lr': 1e-8,
    },
    
    # the directory of all data, including training, validation, and testing data
    'DATA_DIR': './food_data/',

    # the path to label2name.txt
    'LABEL2NAME_PATH': './food_data/label2name.txt',
    
    # the filename of the output csv-file
    'OUTPUT_CSV_FILENAME': 'submission.csv',
    
    # the file-paths of all four sample submission csv-files
    'SAMPLE_SUBMISSION_FILEPATHS': {
        'main': './food_data/testcase/sample_submission_main_track.csv',
        'freq': './food_data/testcase/sample_submission_freq_track.csv',
        'comm': './food_data/testcase/sample_submission_comm_track.csv',
        'rare': './food_data/testcase/sample_submission_rare_track.csv',
    },
    
    # the directory of the saved models
    'MODEL_SAVE_DIR': './saved_models',
    
    # the path of the pretrained word embedding model
    # 'PRETRAINED_WORD_EMBEDDING_MODEL_PATH': './y_360W_cbow_2D_300dim_2020v1.bin',
    'PRETRAINED_WORD_EMBEDDING_MODEL_PATH': './tmunlp_1.6B_WB_300dim_2020v1.bin',
    
    # the transforms for training (data augmentation)
    'TR_TFMS': transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.22, hue=0),
        # transforms.RandomAffine(5, translate=(0.13, 0.13), scale=(0.74, 1.32), fillcolor=(255, 255, 255), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ]),
    
    # the transforms for validation and testing
    'VL_AND_TE_TFMS': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}


label2name_dict = dict()
with open(cfg['LABEL2NAME_PATH'], 'r') as f:
    for line in f.read().splitlines():
        num_lbl, fcr, txt_lbl = line.strip().split(' ')
        label2name_dict[int(num_lbl)] = {'fcr': fcr, 'name': txt_lbl}
# label2name_dict

import fasttext
import fasttext.util

if not os.path.exists('./cc.zh.300.bin'):
    fasttext.util.download_model('zh', if_exists='ignore')
ft = fasttext.load_model('cc.zh.300.bin')



def _calc_cos_sim(vec_1, vec_2, eps=1e-25):
    return (np.dot(vec_1, vec_2) + eps) / ((np.linalg.norm(vec_1) * np.linalg.norm(vec_2)) + eps)

def _calc_words_sim(word_1, word_2):
    sim_12 = np.mean([np.max([_calc_cos_sim(ft[ch_1], ft[ch_2]) for ch_2 in word_2]) for ch_1 in word_1])
    sim_21 = np.mean([np.max([_calc_cos_sim(ft[ch_1], ft[ch_2]) for ch_1 in word_1]) for ch_2 in word_2])
    weighting_ratio = len(word_1) / (len(word_1) + len(word_2))
    sim = weighting_ratio * sim_12 + (1. - weighting_ratio) * sim_21
    return sim

# give it a try
for num_lbl_1 in range(cfg['N_CLS']):
    name_1 = label2name_dict[num_lbl_1]['name']
    if '飯' in name_1:
    # if '蝦' in name_1:
    # if '湯' in name_1:
        calculated = []
        for num_lbl_2 in range(cfg['N_CLS']):
            name_2 = label2name_dict[num_lbl_2]['name']
            sim = _calc_words_sim(name_1, name_2)
            calculated.append((sim, name_1, name_2, num_lbl_1, num_lbl_2))
        
        calculated = sorted(calculated, key=lambda x: x[0])
        buxiang, xiang = [], []
        for sim, name_1, name_2, _num_lbl_1, _num_lbl_2 in calculated:
            # print(f'| {name_1}[{_num_lbl_1}] | {name_2}[{_num_lbl_2}] | {sim:.4f} |')
            if sim < .3:
                buxiang.append(_num_lbl_2)
            if sim > .8:
                xiang.append(_num_lbl_2)
        break



# the customed dataset
class FoodDataset(Dataset):
    def __init__(self, tfm, split):
        assert split in ('train', 'val', 'test'), '"split" must be "train", "val", or "test"'

        self.__data_root = os.path.join(cfg['DATA_DIR'], split)
        self.__split = split
        self.__tfm = tfm

        # when testing: collect the filenames only
        self.__all_data = []
        if self.__split == 'test':
            self.__all_data.extend([img_filename for img_filename in sorted(os.listdir(self.__data_root)) if img_filename.endswith('.jpg')])
        # when training or validation: collect the file-paths and the respective labels
        else:
            for cls_idx in range(cfg['N_CLS']):
                self.__all_data.extend([
                    (os.path.join(self.__data_root, f'{cls_idx}', img_filename), cls_idx) for img_filename in os.listdir(os.path.join(self.__data_root, f'{cls_idx}')) if img_filename.endswith('.jpg')
                ])
            random.shuffle(self.__all_data)
        
        # the length of the dataset
        self.__len = len(self.__all_data)
        print(self.__split, self.__len)

    def get_labels(self):
        return  [lbl for _, lbl in self.__all_data]

    def __getitem__(self, idx):
        # when testing: return the image-id and the image
        if self.__split == 'test':
            filename = self.__all_data[idx]
            img_id = filename[:filename.rindex('.')]
            img = Image.open(os.path.join(self.__data_root, filename)).convert('RGB')
            img = self.__tfm(img)
            return img_id, img
        # when training or validation: return the image and the respective label
        else:
            file_path, lbl = self.__all_data[idx]
            img = Image.open(file_path).convert('RGB')
            img = self.__tfm(img)
            return img, lbl

    def __len__(self):
        return self.__len





# the data-loader for training
tr_dataset = FoodDataset(tfm=cfg['TR_TFMS'], split='train')
tr_loader = DataLoader(tr_dataset, batch_size=cfg['BATCH_SZ'], sampler=ImbalancedDatasetSampler(tr_dataset), num_workers=0, pin_memory=True)
# for validation
vl_dataset = FoodDataset(tfm=cfg['VL_AND_TE_TFMS'], split='val')
vl_loader = DataLoader(vl_dataset, batch_size=cfg['BATCH_SZ'], shuffle=False, num_workers=0, pin_memory=True)
# for testing
te_dataset = FoodDataset(tfm=cfg['VL_AND_TE_TFMS'], split='test')
te_loader = DataLoader(te_dataset, batch_size=cfg['BATCH_SZ'], shuffle=False, num_workers=0, pin_memory=True)





model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    img_size=224,
    num_classes=cfg['N_CLS'],
).to(cfg['DEVICE'])

print(model)


def _obtain_text_label_loss_term(gts):
    batch_sz = gts.shape[0]

    former_gts = gts[:batch_sz // 2]
    latter_gts = gts[batch_sz // 2:][:len(former_gts)]
    
    losses = [
        _calc_words_sim(
            label2name_dict[gt_1.item() if hasattr(gt_1, 'item') else gt_1]['name'],
            label2name_dict[gt_2.item() if hasattr(gt_2, 'item') else gt_2]['name'],
        )
        for gt_1, gt_2 in zip(former_gts, latter_gts) if gt_1 != gt_2
    ]
    # print(losses); input()
    return np.mean(losses)

# fan = [23] * len(buxiang)
# print(_obtain_text_label_loss_term(np.array(fan + buxiang)))
# fan = [23] * len(xiang)
# print(_obtain_text_label_loss_term(np.array(fan + xiang)))



import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1, 
                 reduction="mean", weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing   = smoothing
        self.reduction = reduction
        self.weight    = weight

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
         if self.reduction == 'sum' else loss

    def linear_combination(self, x, y):
        return self.smoothing * x + (1 - self.smoothing) * y

    def forward(self, preds, target): # if preds: [batch_size, num_classes], target: [barch_size, num_classes]
        assert 0 <= self.smoothing < 1

        if self.weight is not None:
            self.weight = self.weight.to(preds.cfg['DEVICE'])

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1))
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction, weight=self.weight
        )
        return self.linear_combination(loss / n, nll)




def PairwiseConfusion(features, target):
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        return 0
    batch_left = features[:int(0.5*batch_size)]
    batch_right = features[int(0.5*batch_size):]

    target_left = target[:int(0.5*batch_size)]
    target_right = target[int(0.5*batch_size):]

    target_mask_t = torch.eq(target_left, target_right)
    target_mask = ~target_mask_t
    target_mask = target_mask.type(torch.cuda.FloatTensor)
    number = target_mask.sum()
    loss  = (torch.norm((batch_left - batch_right).abs(),2, 1) * target_mask).sum() / number

    return loss




import math
from typing import Callable, Iterable, Optional, Tuple, Union

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)




label_file = open(cfg['LABEL2NAME_PATH'])
label_mapping = [line.split(' ')[1] for line in label_file.readlines()]

def count_acc_category(acc_list):
    c_accs, f_accs, r_accs = [], [], []
    for label, acc in acc_list:
        if label_mapping[label] == 'c':
            c_accs.append(acc)
        elif label_mapping[label] == 'f':
            f_accs.append(acc)
        elif label_mapping[label] == 'r':
            r_accs.append(acc)
    return np.array(c_accs).mean(), np.array(f_accs).mean(), np.array(r_accs).mean()



num_epochs = 12
save_per_iters = 500
output_path = './checkpoints_2'
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print("create output_dir: ", output_path)

#criterion = nn.CrossEntropyLoss().to(cfg['DEVICE']EVICE'])
smoothing = 0.1
criterion = LabelSmoothingLoss(smoothing=smoothing).to(cfg['DEVICE'])

#origin_lr=0.1
origin_lr=1e-5
momentum=0.9
weight_decay=1e-6
#optimizer = torch.optim.SGD(model.parameters(), origin_lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr=origin_lr, weight_decay=weight_decay)

factor = 0.9
patience = 1
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)




start_epoch = 0
best_valid_accu = 0.0
iters = 0

Train_Loss_Iter = []
Train_Accu_Iter = []

Valid_Loss_Iter = []
Valid_Accu_Iter = []

Valid_Loss_Epoch = []
Valid_Accu_Epoch = []



model.train()

for epoch in range(start_epoch, num_epochs):

    # adjust_learning_rate(optimizer, epoch, origin_lr)

    # train for one epoch
    # end = time.time()
    for i, (images, target) in enumerate(tr_loader):
        # switch to train mode 
        model.train()
        # measure data loading time
        # data_time.update(time.time() - end)
        images = images.to(cfg['DEVICE'])
        target = target.to(cfg['DEVICE'])

        # compute output
        output = model(images)
        loss = criterion(output, target) + .5 * _obtain_text_label_loss_term(target)
        #pairwise_confusion = PairwiseConfusion(output, target)
        #total_loss = loss + pairwise_confusion
        acc = 100.0 * (output.argmax(dim=-1) == target).float().mean()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        #total_loss.backward()
        loss.backward()
        optimizer.step()

        Train_Loss_Iter.append(loss.data.cpu().numpy())
        Train_Accu_Iter.append(acc.data.cpu().numpy())

        current_lr = optimizer.param_groups[0]["lr"]
        # print ('epoch: %d, iters: %d, train_accu: %.4f, train_loss: %.4f, lr: %.8f' \
        #           % (epoch+1, iters+1, acc, loss, current_lr))
        print(f'====== | [TR] | EP = {epoch + 1:03d}/{num_epochs:03d} | ITER = {iters + 1:04d} | LOSS = {loss:.4f} | ACC = {acc:.4f} | LR = {current_lr:.8f} | ======')

        if (iters+1) % save_per_iters == 0:
            # progress.print(i)
            model_path = "Iter" + str(iters+1) + "_model.pt"
            torch.save({
                'epoch': epoch + 1,
                'iters': iters + 1,
                'state_dict': model.state_dict(),
                'best_valid_accu': best_valid_accu,
                #'optimizer' : optimizer.state_dict(),
            }, os.path.join(output_path, model_path))

            # I. Calculate Valid Accu/Loss
            valid_loss = 0
            correct = 0
            valid_accu = 0
            predictions = []
            vl_category_list = []
            # switch to evaluate mode
            model.eval()
            with torch.no_grad():
                for i, (images, target) in enumerate(vl_loader):
                    images = images.to(cfg['DEVICE'])
                    target = target.to(cfg['DEVICE'])

                    # compute output
                    output = model(images)
                    loss = criterion(output, target)

                    # self-measurement
                    valid_loss += criterion(output, target).item() # sum up batch loss
                    pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    predictions.extend(output.argmax(dim=-1).cpu().numpy().tolist())

                    vl_category_list += [(target[i].item(), (output.argmax(dim=-1) == target)[i].float().item()) for i in range(output.shape[0])]

                valid_loss /= len(vl_loader.dataset)
                valid_accu = 100. * correct / len(vl_loader.dataset)
                c_acc, f_acc, r_acc = count_acc_category(vl_category_list)
                print ('epoch: %d, iters: %d, valid_accu: %.4f, valid_loss: %.4f, train_accu: %.4f, train_loss: %.4f' \
                  % (epoch+1, iters+1, valid_accu, valid_loss, acc, loss))
                print ('c_acc: %.4f, f_acc: %.4f, r_acc: %.4f' % (c_acc, f_acc, r_acc))

                Valid_Loss_Iter.append(valid_loss)
                Valid_Accu_Iter.append(valid_accu)

                scheduler.step(valid_loss)

                if valid_accu > best_valid_accu:
                    best_valid_accu = valid_accu
                    print("Save Best Valid Accu")
                    model_path = "model_best.pt"
                    torch.save({
                        'epoch': epoch + 1,
                        'iters': iters + 1,
                        'state_dict': model.state_dict(),
                        'best_valid_accu': best_valid_accu,
                        #'optimizer' : optimizer.state_dict(),
                    }, os.path.join(output_path, model_path))
                
                model_path = "model_last.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'iters': iters + 1,
                    'state_dict': model.state_dict(),
                    'best_valid_accu': best_valid_accu,
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'Train_Loss_Iter': Train_Loss_Iter,
                    'Train_Accu_Iter': Train_Accu_Iter,

                    'Valid_Loss_Iter': Valid_Loss_Iter,
                    'Valid_Accu_Iter': Valid_Accu_Iter,

                    'Valid_Loss_Epoch': Valid_Loss_Epoch,
                    'Valid_Accu_Epoch': Valid_Accu_Epoch,
                }, os.path.join(output_path, model_path))
                # if int(math.floor(alid_accu)) == 94:
        #scheduler.step(iters)
        iters = iters + 1
    # train(tr_loader, model, criterion, optimizer, epoch, args)

    model_path = "Epoch" + str(epoch+1) + "_model.pt"
    torch.save({
        'epoch': epoch + 1,
        'iters': iters + 1,
        'state_dict': model.state_dict(),
        'best_valid_accu': best_valid_accu,
        #'optimizer' : optimizer.state_dict(),
    }, os.path.join(output_path, model_path))

    # evaluate on validation set
    valid_loss = 0
    correct = 0
    valid_accu = 0
    predictions = []
    vl_category_list = []
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(vl_loader):
            images = images.to(cfg['DEVICE'])
            target = target.to(cfg['DEVICE'])

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # self-measurement
            valid_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            predictions.extend(output.argmax(dim=-1).cpu().numpy().tolist())

            vl_category_list += [(target[i].item(), (output.argmax(dim=-1) == target)[i].float().item()) for i in range(output.shape[0])]

        valid_loss /= len(vl_loader.dataset)
        valid_accu = 100. * correct / len(vl_loader.dataset)
        # print ('epoch: %d, iters: %d, valid_accu: %.4f, valid_loss: %.4f' \
        #   % (epoch+1, iters+1, valid_accu, valid_loss))
        # print ('c_acc: %.4f, f_acc: %.4f, r_acc: %.4f' % (c_acc, f_acc, r_acc))
        print(f'\t\t====== | [VL] | EP = {epoch + 1:03d}/{num_epochs:03d} | ITER = {iters + 1:04d} | LOSS = {valid_loss:.4f} | ACC = {valid_accu:.4f} | ======')
        print(f'\t\t====== | [VL] | FREQ-ACC = {f_acc:.4f} | COMM-ACC = {c_acc:.4f} | RARE-ACC = {r_acc:.4f} | ======')
        Valid_Loss_Epoch.append(valid_loss)
        Valid_Accu_Epoch.append(valid_accu)

        # scheduler.step(valid_loss)

        if valid_accu > best_valid_accu:
            best_valid_accu = valid_accu
            print("Save Best Valid Accu")
            model_path = "model_best.pt"
            torch.save({
                'epoch': epoch + 1,
                'iters': iters + 1,
                'state_dict': model.state_dict(),
                'best_valid_accu': best_valid_accu,
                'optimizer' : optimizer.state_dict(),
            }, os.path.join(output_path, model_path))





model_path = "model_last.pt"
torch.save({
    'epoch': epoch + 1,
    'iters': iters + 1,
    'state_dict': model.state_dict(),
    'best_valid_accu': best_valid_accu,
    'optimizer' : optimizer.state_dict(),
    'scheduler' : scheduler.state_dict(),
}, os.path.join(output_path, model_path))