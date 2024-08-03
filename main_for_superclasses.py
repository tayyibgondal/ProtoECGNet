import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import argparse
import re
import pandas as pd
import numpy as np

from helpers import makedir
import model
import push
import train_test_inference as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

import wandb
wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-base', nargs=1, type=str, default='vgg19') 
parser.add_argument('-experiment_run', nargs=1, type=str, default='0') 
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

# book keeping namings and code
from settings import img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type
base_architecture = args.base[0]
experiment_run = args.experiment_run[0]

base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)
model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_test_inference.py'), dst=model_dir)

from train_test_inference import log_to_file_and_console
log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
log = log_to_file_and_console  # CHANGE THE LOG FUNCTION

img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size
# ---------------------------------------------------------------
# Updated data loader code
from settings import scp_statements_path, ptb_database_file_path, data_dir, num_train_examples, num_test_examples
from dataset_class_for_superclasses import ECGImageDataset

# Define transformations
img_size = 224  # or whatever size you want
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize,
])

# Function to create a subset of the dataset
def create_subset(dataset, num_examples):
    # Ensure num_examples doesn't exceed the dataset length
    num_examples = min(len(dataset), num_examples)
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

# Initialize dataset and dataloader for testing
test_dataset = ECGImageDataset(scp_statements_path, ptb_database_file_path, img_dir, transform=transform, test=True)
# Initialize dataset and dataloader for training
train_dataset = ECGImageDataset(scp_statements_path, ptb_database_file_path, img_dir, transform=transform, test=False)

if num_train_examples is not None:
    train_subset = create_subset(train_dataset, num_train_examples)
else: 
    train_subset = train_dataset

if num_test_examples is not None:
    test_subset = create_subset(test_dataset, num_test_examples)
else:
    test_subset = test_dataset

# Create data loaders for the subsets
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False)
test_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=True, num_workers=4, pin_memory=False)

# ---------------------------------------------------------------
# normalize = transforms.Normalize(mean=mean,
#                                  std=std)

# all datasets
# train set
# train_dataset = datasets.ImageFolder(
#     train_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#         normalize,
#     ]))
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=train_batch_size, shuffle=True,
#     num_workers=4, pin_memory=False)
# # push set
# train_push_dataset = datasets.ImageFolder(
#     train_push_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#     ]))
# train_push_loader = torch.utils.data.DataLoader(
#     train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
#     num_workers=4, pin_memory=False)
# # test set
# test_dataset = datasets.ImageFolder(
#     test_dir,
#     transforms.Compose([
#         transforms.Resize(size=(img_size, img_size)),
#         transforms.ToTensor(),
#         normalize,
#     ]))
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=test_batch_size, shuffle=False,
#     num_workers=4, pin_memory=False)

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
log('training set size: {0}'.format(len(train_loader.dataset)))
# log('push set size: {0}'.format(len(train_push_loader.dataset)))
# log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)
#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.to('cuda')
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size
joint_optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)

from settings import warm_optimizer_lrs
warm_optimizer_specs = \
[{'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
]
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, target_auroc, target_f1

# train the model
log('start training')
import copy

run = wandb.init(
    # Set the project where this run will be logged
    project="ExplainableECGModels",
    # Track hyperparameters and run metadata
    config={
        "backend": base_architecture,
        "experiment_run": experiment_run
    },
)

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    if epoch < num_warm_epochs:
        tnt.warm_only(model=ppnet_multi, log=log)
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)
    else:
        tnt.joint(model=ppnet_multi, log=log)
        joint_lr_scheduler.step()
        _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                      class_specific=class_specific, coefs=coefs, log=log)

    accu, f1, auroc = tnt.test(model=ppnet_multi, dataloader=test_loader, 
                    class_specific=class_specific, log=log)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', f1=f1,
                                target_f1=target_f1, auroc=auroc, target_auroc=target_auroc, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_loader, # pytorch dataloader (must be unnormalized in [0,1])   # CHANGE TRAIN_PUSH_LOADER TO TRAIN_LOADER
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, f1, auroc = tnt.test(model=ppnet_multi, dataloader=test_loader,  
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', f1=f1,
                                target_f1=target_f1, auroc=auroc, target_auroc=target_auroc, log=log)
 
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                              class_specific=class_specific, coefs=coefs, log=log)
                accu, f1, auroc = tnt.test(model=ppnet_multi, dataloader=test_loader, 
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', f1=f1, target_f1=target_f1, auroc=auroc, target_auroc=target_auroc, log=log)
   
logclose()

