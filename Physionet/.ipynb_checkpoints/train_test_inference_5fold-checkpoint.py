from helpers import list_of_distances, make_one_hot

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from sklearn.metrics import roc_auc_score

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import wandb
from tqdm import tqdm

from settings import img_size, log_dir, prototype_shape, num_prototypes_for_each_class

def log_to_file_and_console(message, logfile=None):
    print(message)
    
    if logfile is not None:
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        with open(logfile, 'a') as f:
            f.write(message + '\n')

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=log_to_file_and_console, fold=None):
    is_train = optimizer is not None
    if is_train:
        load_desc = 'train'
    else:
        load_desc = 'test'
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0

    all_labels = []
    all_scores = []

    # Instantiate the loss function
    criterion = torch.nn.BCELoss()

    for i, (image, label) in enumerate(tqdm(dataloader, desc=f"Processing {load_desc} batches")):
        input = image.cuda()
        target = label.cuda().float()

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input)

            cross_entropy = criterion(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # Tells which prototypes should be activated for all examples in the batch 
                prototypes_of_correct_class = target.repeat_interleave(num_prototypes_for_each_class, dim=1) # batch_size, num_prototypes
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            n_examples += target.size(0)
            n_batches += 1

            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

            all_labels.extend(target.cpu().numpy())
            all_scores.extend(output.detach().cpu().numpy())

        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del min_distances

    end = time.time()

    log(f'\ttime-fold{fold}: \t{end - start}')
    log(f'\tcross ent-fold{fold}: \t{total_cross_entropy / n_batches}')
    log(f'\tcluster-fold{fold}: \t{total_cluster_cost / n_batches}')
    if class_specific:
        log(f'\tseparation-fold{fold}:\t{total_separation_cost / n_batches}')
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log(f'\tp dist pair-fold{fold}: \t{p_avg_pair_dist.item()}')

    # Compute AUROC for multilabel classification
    all_labels_np = np.array(all_labels)
    all_scores_np = np.array(all_scores)
    # Compute the multilabel ROC AUC score
    auroc = roc_auc_score(all_labels_np, all_scores_np, average="macro")
    log(f'\tauroc-fold{fold}: \t{auroc}') 

    return auroc, total_cross_entropy / n_batches, total_cluster_cost / n_batches, total_separation_cost / n_batches, l1, p_avg_pair_dist.item(), fold


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print, fold=None):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    auroc, ce, cc, sc, l1, p_pair_dist, fold = _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer, class_specific=class_specific, coefs=coefs, log=log, fold=fold)

    # Log metrics to WandB
    wandb.log({
        f"Train AUROC fold-{fold}": auroc,
        f"Train Cross Entropy fold-{fold}": ce,
        f"Train Cluster Cost fold-{fold}": cc,
        f"Train Separation Cost fold-{fold}": sc if class_specific else 0,
        f"Train L1 Norm fold-{fold}": l1,
        f"Train P Dist Pair fold-{fold}": p_pair_dist
    })
    
    return auroc

def test(model, dataloader, class_specific=False, log=print, fold=None):
    log('\ttest')
    model.eval()
    auroc, ce, cc, sc, l1, p_pair_dist, fold = _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log, fold=fold)

    # Log metrics to WandB
    wandb.log({
        f"Test AUROC fold-{fold}": auroc,
        f"Test Cross Entropy fold-{fold}": ce,
        f"Test Cluster Cost fold-{fold}": cc,
        f"Test Separation Cost fold-{fold}": sc if class_specific else 0,
        f"Test L1 Norm fold-{fold}": l1,
        f"Test P Dist Pair fold-{fold}": p_pair_dist
    })
    
    return auroc


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
