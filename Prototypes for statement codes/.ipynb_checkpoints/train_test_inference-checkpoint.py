from helpers import list_of_distances, make_one_hot

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score

import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import wandb

from settings import base_architecture, experiment_run, img_size, log_dir, prototype_shape, num_prototypes_for_each_class

# Construct the log file name
log_file_name = os.path.join(log_dir, 'results-' + base_architecture + '-' + experiment_run + '.txt')

def log_to_file_and_console(message, logfile=log_file_name):
    print(message)
    
    # Ensure the log directory exists
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    with open(logfile, 'a') as f:
        f.write(message + '\n')

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=log_to_file_and_console):
    is_train = optimizer is not None
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
    criterion = torch.nn.BCEWithLogitsLoss()

    for i, (image, labels) in enumerate(dataloader):
        input = image.cuda() # (batch_size, 3, 224, 224)
        
        scp_labels = labels[1].cuda() # (batch_size, 71)
        label = labels[0]  # (batch_size, 5)

        target = label.cuda().float()  # (batch_size, num_classes) --> one hot labels for all examples


        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input) # Output: (b, num_classes)   Min_dis: (b, num_pro)

            cross_entropy = criterion(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # Tells which prototypes should be activated for all examples in the batch 
                prototypes_of_correct_class = scp_labels.repeat_interleave(num_prototypes_for_each_class, dim=1) # batch_size, num_prototypes

                # shape of max_dist-min_dist = (batch, num_protos)
                # shape of proto_corr_class = (batch, num_protos)
                # (max_dis-min_dis) * proto_of_corr_class --> extract the distances of only those prototypes which are activated, for a particular example in the batch
                # torch.max will then extract the maximum inverted distanced prototype (or activated prototype) out of all the prototypes 
                # for multi labeled problem, we would do sum rather than max.
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)  # (batch_size,)
                # inverted_distance represents the inverted_distance of that prototype, which corresponds to the true label.
                
                cluster_cost = torch.mean(max_dist - inverted_distances) 

                prototypes_of_wrong_class = 1 - prototypes_of_correct_class  # (batch_size, num_protos)
                # close prototypes of other classes, will be selected
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1) # (batch_size, num_protos)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # -----------------------------
                # if use_l1_mask:
                #     l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                #     l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)  # the norm of those weights which correspond to wrong class, should be kept in bounds
                # else:
                #     l1 = model.module.last_layer.weight.norm(p=1)
                # -----------------------------
                l1 = model.module.last_layer.weight.norm(p=1)
                # -----------------------------

            else:
                min_distance, _ = torch.min(min_distances, dim=1)  # (batch, 1)
                cluster_cost = torch.mean(min_distance)
                # -----------------------------
                l1 = model.module.last_layer.weight.norm(p=1)
                # -----------------------------


            n_examples += target.size(0)
            n_batches += 1

            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()

            # Append to all_labels and all_scores
            all_labels.extend(target.cpu().numpy())
            all_scores.extend(torch.sigmoid(output).detach().cpu().numpy())  # Apply sigmoid for multilabel

        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                    # loss = (coefs['crs_ent'] * cross_entropy
                    #       + coefs['clst'] * cluster_cost
                    #       + coefs['sep'] * separation_cost)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
                    # loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost
            else:
                if coefs is not None:
                    # loss = (coefs['crs_ent'] * cross_entropy
                    #       + coefs['clst'] * cluster_cost)
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    # loss = cross_entropy + 0.8 * cluster_cost
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del min_distances

    end = time.time()

    log(f'\ttime: \t{end - start}')
    log(f'\tcross ent: \t{total_cross_entropy / n_batches}')
    log(f'\tcluster: \t{total_cluster_cost / n_batches}')
    if class_specific:
        log(f'\tseparation:\t{total_separation_cost / n_batches}')
    # log(f'\tl1: \t\t{model.module.last_layer.weight.norm(p=1).item()}')
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log(f'\tp dist pair: \t{p_avg_pair_dist.item()}')

    # Compute AUROC for multilabel classification
    all_labels_np = np.array(all_labels)
    all_scores_np = np.array(all_scores)
    # Compute the multilabel ROC AUC score
    auroc = roc_auc_score(all_labels_np, all_scores_np, average="macro")
    log(f'\tAUROC: {auroc:.4f}')

    return auroc, total_cross_entropy / n_batches, total_cluster_cost / n_batches, total_separation_cost / n_batches, p_avg_pair_dist.item()


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    auroc, ce, cc, sc, p_pair_dist = _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)

    # Log metrics to WandB
    wandb.log({
        "Train AUROC": auroc,
        "Train Cross Entropy": ce,
        "Train Cluster Cost": cc,
        "Train Separation Cost": sc if class_specific else 0,
        "Train P Dist Pair": p_pair_dist
    })
    
    return auroc

def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    auroc, ce, cc, sc, p_pair_dist = _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)

    # Log metrics to WandB
    wandb.log({
        "Test AUROC": auroc,
        "Test Cross Entropy": ce,
        "Test Cluster Cost": cc,
        "Test Separation Cost": sc if class_specific else 0,
        "Test P Dist Pair": p_pair_dist
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