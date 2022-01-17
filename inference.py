########################################################################
##### I. Import Packages
########################################################################
import os
import sys
import csv
import random
import copy

import numpy as np
import pandas as pd
from PIL import Image
import timm
import ttach as tta
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader

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

########################################################################
##### II. Load args
########################################################################
data_dir, output_dir, model_dir = sys.argv[1], sys.argv[2], sys.argv[3]

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

########################################################################
##### III. Dataset and DataLoader
########################################################################
class FoodDataset(Dataset):
    def __init__(self, tfm, split):
        assert split in ('train', 'val', 'test'), '"split" must be "train", "val", or "test"'

        self.__data_root = os.path.join(data_dir, split)
        self.__split = split
        self.__tfm = tfm

        # when testing: collect the filenames only
        self.__all_data = []
        if self.__split == 'test':
            self.__all_data.extend([img_filename for img_filename in sorted(os.listdir(self.__data_root)) if img_filename.endswith('.jpg')])
        # when training or validation: collect the file-paths and the respective labels
        else:
            for cls_idx in range(1000):
                self.__all_data.extend([
                    (os.path.join(self.__data_root, f'{cls_idx}', img_filename), cls_idx) for img_filename in os.listdir(os.path.join(self.__data_root, f'{cls_idx}')) if img_filename.endswith('.jpg')
                ])
            random.shuffle(self.__all_data)
        
        # the length of the dataset
        self.__len = len(self.__all_data)
        print(self.__split, self.__len)

    def get_labels(self):
        return  [lbl for file_path, lbl in self.__all_data]

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

IMG_SIZE = 224
batch_size = 32
test_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
test_set = FoodDataset(tfm=test_tfm, split='test')
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

######################################################################
##### IV. Model
######################################################################
num_classes = 1000
model = timm.create_model('vit_base_patch16_224', pretrained=True,
        img_size=IMG_SIZE,
        num_classes=num_classes)

######################################################################
##### V. Evaluation
######################################################################

def do_testing_w_single_ckpt(model, ckpt_path, te_loader, device):
    loaded = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in loaded:
        loaded = loaded['state_dict']
        if 'head.0.weight' in loaded:
            loaded['head.weight'] = copy.deepcopy(loaded['head.0.weight'])
            del loaded['head.0.weight']
        if 'head.0.bias' in loaded:
            loaded['head.bias'] = copy.deepcopy(loaded['head.0.bias'])
            del loaded['head.0.bias']
    
    model.load_state_dict(loaded)
    model.eval()
    # model = model.to(device)
    print(ckpt_path)
    
    tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.hflip_transform()).to(device)

    # output_dict = dict()
    logits_list = []
    img_ids_list = []
    # switch to evaluate mode
    tta_model.eval()
    with torch.no_grad():
        for i, (img_ids, images) in enumerate(te_loader):
            images = images.to(device)
        
            # compute output
            logits = tta_model(images)
            # outputs = logits.argmax(dim=-1).cpu().tolist()
            # for img_id, output in zip(img_ids, outputs):
            #     output_dict[img_id] = output
            
            logits_list.extend(logits.detach().cpu().tolist())
            img_ids_list.extend(img_ids)
    return logits_list, img_ids_list

# obtain all checkpoints used to do ensemble
ckpt_filename_list = [model_filename for model_filename in os.listdir(model_dir) if model_filename.endswith('.pt') or model_filename.endswith('.ckpt')]

# do ensemble
ensembled_logits_list = None
final_img_ids_list = None
for ckpt_filename in ckpt_filename_list:
    logits_list, img_ids_list = do_testing_w_single_ckpt(model, os.path.join(model_dir, ckpt_filename), test_loader, device)
    if ensembled_logits_list is None:
        ensembled_logits_list = np.array(logits_list)
    else:
        ensembled_logits_list += np.array(logits_list)
    if final_img_ids_list is None:
        final_img_ids_list = img_ids_list
print(ensembled_logits_list.shape)

# generate the final outputs
output_dict = dict()
for ensembled_logits, img_id in zip(ensembled_logits_list, final_img_ids_list):
    output_dict[img_id] = ensembled_logits.argmax(axis=-1).tolist()
print(len(output_dict))

######################################################################
##### VI. CSV Generator
######################################################################
def csv_generator(in_path, output_dict, out_path):
    with open(in_path, 'r') as f_sample:
        sample_rows = list(csv.reader(f_sample, delimiter=','))[1:]
        print(sample_rows)
        # write the csv-file of this track
        with open(out_path, 'w') as f_out:
            # write the first row
            f_out.write('image_id,label\n')
            # write the outputs
            for sample_row in sample_rows:
                img_id = sample_row[0]
                f_out.write(f'{img_id},{output_dict[img_id]}\n')

######################################################################
##### VII. Generate CSV Output
######################################################################
main_track_in_path = os.path.join(data_dir, "testcase/sample_submission_main_track.csv")
freq_track_in_path = os.path.join(data_dir, "testcase/sample_submission_freq_track.csv")
comm_track_in_path = os.path.join(data_dir, "testcase/sample_submission_comm_track.csv")
rare_track_in_path = os.path.join(data_dir, "testcase/sample_submission_rare_track.csv")

main_track_out_path = os.path.join(output_dir, "main.csv")
freq_track_out_path = os.path.join(output_dir, "freq.csv")
comm_track_out_path = os.path.join(output_dir, "comm.csv")
rare_track_out_path = os.path.join(output_dir, "rare.csv")

csv_generator(main_track_in_path, output_dict, main_track_out_path)
csv_generator(freq_track_in_path, output_dict, freq_track_out_path)
csv_generator(comm_track_in_path, output_dict, comm_track_out_path)
csv_generator(rare_track_in_path, output_dict, rare_track_out_path)