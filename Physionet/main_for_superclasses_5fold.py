import os
import shutil
import torch
import torch.utils.data
import torchvision.transforms as transforms
import argparse
import re
import pandas as pd
import numpy as np
from helpers import makedir
import model_for_superclasses as model
import push
import train_test_inference_5fold as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function
from sklearn.model_selection import KFold

import wandb
wandb.login()

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-base', nargs=1, type=str, default='vgg19') 
parser.add_argument('-experiment_run', nargs=1, type=str, default='0') 
parser.add_argument("-run_name", type=str, default="default_run_name", help="Name of the W&B run")
parser.add_argument("-fold", type=int, default=None, help="Specify the fold number to train")

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# Bookkeeping and setup
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
shutil.copy(src=os.path.join(os.getcwd(), 'train_test_inference_5fold.py'), dst=model_dir)

from train_test_inference_5fold import log_to_file_and_console
log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
log = log_to_file_and_console  # CHANGE THE LOG FUNCTION
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

run = wandb.init(
        project="ExplainableECGModels",
        name=args.run_name,
        config={
            "backend": base_architecture,
            "experiment_run": experiment_run,
            "fold": args.fold
        },
    )

# Load the data
from settings import train_batch_size, test_batch_size, num_examples
from settings import train_information, test_information, num_train_examples, num_test_examples, img_size
from dataset_class import ECGImageDataset

# Define transformations
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize,
])

def create_subset(dataset, num_examples):
    num_examples = min(len(dataset), num_examples)
    indices = np.random.choice(len(dataset), num_examples, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

dataset = ECGImageDataset('dataset-100.json', transform=transform)

if num_examples is not None:
    subset = create_subset(dataset, num_examples)
else:
    subset = dataset

# Setup K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# Get the specified fold number from the command line argument
specified_fold = args.fold

# Cross-validation loop
for fold, (train_ids, test_ids) in enumerate(kfold.split(subset)):
    if specified_fold is not None and fold != specified_fold:
        continue  # Skip folds that don't match the specified fold

    print(f"Training on FOLD {fold}")
    print("--------------------------------")
    
    # Save train and test indices for this fold
    fold_dir = os.path.join(model_dir, f"fold_{fold}")
    makedir(fold_dir)
    np.save(os.path.join(fold_dir, 'train_indices.npy'), train_ids)
    np.save(os.path.join(fold_dir, 'test_indices.npy'), test_ids)

    # Sample elements randomly from the dataset to create train and test subsets
    train_subsampler = torch.utils.data.Subset(subset, train_ids)
    test_subsampler = torch.utils.data.Subset(subset, test_ids)

    # Define data loaders for training and testing data in this fold
    train_loader = torch.utils.data.DataLoader(train_subsampler, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_subsampler, batch_size=test_batch_size, shuffle=False, num_workers=4, pin_memory=False)

    log(f'training set size: {len(train_loader.dataset)}')
    log(f'batch size: {train_batch_size}')
    
    # Construct the model
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                  pretrained=True, img_size=img_size,
                                  prototype_shape=prototype_shape,
                                  num_classes=num_classes,
                                  prototype_activation_function=prototype_activation_function,
                                  add_on_layers_type=add_on_layers_type)
    # if prototype_activation_function == 'linear':
    #    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
    ppnet = ppnet.to('cuda')
    ppnet_multi = torch.nn.DataParallel(ppnet)
    class_specific = True

    # Define optimizers
    from settings import joint_optimizer_lrs, joint_lr_step_size
    joint_optimizer_specs = [
        {'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3},
        {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
    
    from settings import warm_optimizer_lrs
    warm_optimizer_specs = [
        {'params': ppnet.add_on_layers.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
        {'params': ppnet.prototype_vectors, 'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    
    from settings import last_layer_optimizer_lr
    last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    
    # Weighting of different training losses
    from settings import coefs
    
    # Number of training epochs, number of warm epochs, push start epoch, push epochs
    from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, target_auroc
    
    log('Start training')
    
    for epoch in range(num_train_epochs):
        print(f'Epoch {epoch+1}/{num_train_epochs} for fold {fold}')
    
        if epoch < num_warm_epochs:
            tnt.warm_only(model=ppnet_multi, log=log)
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, fold=fold)
        else:
            tnt.joint(model=ppnet_multi, log=log)
            joint_lr_scheduler.step()
            _ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, fold=fold)
    
        auroc = tnt.test(model=ppnet_multi, dataloader=test_loader, 
                        class_specific=class_specific, log=log, fold=fold)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=f'fold_{fold}-{epoch}nopush', auroc=auroc, target_auroc=target_auroc, log=log)
    
        if epoch >= push_start and epoch in push_epochs:
            push.push_prototypes(
                train_loader,
                prototype_network_parallel=ppnet_multi,
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir,
                epoch_number=epoch,
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log)
            auroc = tnt.test(model=ppnet_multi, dataloader=test_loader,  
                            class_specific=class_specific, log=log, fold=fold)
            save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=f'fold_{fold}-{epoch}push', auroc=auroc, target_auroc=target_auroc, log=log)
    
    fold_results.append((fold, auroc))
    log(f'Fold {fold} AUROC: {auroc}')
    log(f"Fold {fold} results appended to fold_results")

# Save fold results
fold_results_file = os.path.join('fold_results', f'{specified_fold}fold_results.csv')
pd.DataFrame(fold_results, columns=["Fold", "AUROC"]).to_csv(fold_results_file, index=False)
log("Training completed for specified fold(s).")
logclose()
