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

from settings import base_architecture, experiment_run, img_size, log_dir, prototype_shape, label_index_to_label_text_mapping, num_prototypes_for_each_class

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
    total_avg_separation_cost = 0

    all_labels = []
    all_scores = []

    for i, (image, label) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input)

            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1)

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_batches += 1
            n_correct += (predicted == target).sum().item()

            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

            # Append to all_labels and all_scores
            all_labels.extend(target.cpu().numpy())
            all_scores.extend(output.softmax(dim=1).detach().cpu().numpy())

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
        del predicted
        del min_distances

    end = time.time()

    log(f'\ttime: \t{end - start}')
    log(f'\tcross ent: \t{total_cross_entropy / n_batches}')
    log(f'\tcluster: \t{total_cluster_cost / n_batches}')
    if class_specific:
        log(f'\tseparation:\t{total_separation_cost / n_batches}')
        log(f'\tavg separation:\t{total_avg_separation_cost / n_batches}')
    log(f'\taccu: \t\t{n_correct / n_examples * 100}%')
    log(f'\tl1: \t\t{model.module.last_layer.weight.norm(p=1).item()}')
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log(f'\tp dist pair: \t{p_avg_pair_dist.item()}')

    # Calculate metrics
    all_scores = np.array(all_scores)
    accuracy = accuracy_score(all_labels, [score.argmax() for score in all_scores])
    f1 = f1_score(all_labels, [score.argmax() for score in all_scores], average='weighted')

    auroc = roc_auc_score(all_labels, all_scores, multi_class='ovr', average='weighted')

    log(f'\tAccuracy: {accuracy * 100:.2f}%')
    log(f'\tF1 Score: {f1:.4f}')
    log(f'\tAUROC: {auroc:.4f}')

    return n_correct / n_examples, f1, auroc, total_cross_entropy / n_batches, total_cluster_cost / n_batches, total_separation_cost / n_batches, total_avg_separation_cost / n_batches, model.module.last_layer.weight.norm(p=1).item(), p_avg_pair_dist.item()


def train(model, dataloader, optimizer, class_specific=False, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    accuracy, f1, auroc, ce, cc, sc, avg_sc, l1, p_pair_dist = _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log)

    # Log metrics to WandB
    wandb.log({
        "Train Accuracy": accuracy * 100,
        "Train F1 Score": f1,
        "Train AUROC": auroc,
        "Train Cross Entropy": ce,
        "Train Cluster Cost": cc,
        "Train Separation Cost": sc if class_specific else 0,
        "Train Avg Separation Cost": avg_sc if class_specific else 0,
        "Train L1 Norm": l1,
        "Train P Dist Pair": p_pair_dist
    })
    
    return accuracy, f1, auroc

def test(model, dataloader, class_specific=False, log=print):
    log('\ttest')
    model.eval()
    accuracy, f1, auroc, ce, cc, sc, avg_sc, l1, p_pair_dist = _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)

    # Log metrics to WandB
    wandb.log({
        "Test Accuracy": accuracy * 100,
        "Test F1 Score": f1,
        "Test AUROC": auroc,
        "Test Cross Entropy": ce,
        "Test Cluster Cost": cc,
        "Test Separation Cost": sc if class_specific else 0,
        "Test Avg Separation Cost": avg_sc if class_specific else 0,
        "Test L1 Norm": l1,
        "Test P Dist Pair": p_pair_dist
    })
    
    return accuracy, f1, auroc


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


def inference(model, example_path, class_specific=True, prototype_img_folder=None):
    model.eval()

    # Define transformations
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    # Load and transform the image
    image = Image.open(example_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    output, min_distances = model(input_tensor)
    predicted_class_label = output.data.max(1)[1].cpu()  # get the index of the max log-probability and move to CPU
    predicted_class_label_text = label_index_to_label_text_mapping[predicted_class_label.item()]

    # Print the predicted class
    if predicted_class_label == 0:
        print('Predicted class:', predicted_class_label_text)
    else:
        print('Predicted class:', predicted_class_label_text)

    # Plot the original image
    plt.imshow(image)
    plt.title('Input Image - Predicted Class: ' + predicted_class_label_text)
    plt.show()

    if prototype_img_folder:
        # if class_specific, only show the prototype of the predicted class, otherwise show images of all prototypes
        if class_specific:
            # compute prototype indices range
            start_idx = predicted_class_label.item() * num_prototypes_for_each_class
            end_idx = start_idx + num_prototypes_for_each_class
            
            # Plot prototypes of the predicted class only
            for idx in range(start_idx, end_idx):
                prototype_img_path = f"{prototype_img_folder}/prototype-img{idx}.png"
                prototype_img = plt.imread(prototype_img_path)
                plt.imshow(prototype_img, cmap='gray')
                plt.title(f'Prototype {idx} for class {predicted_class_label.item()}')
                plt.show()
        else:
            # Plot all prototypes
            num_prototypes = model.module.num_prototypes
            for idx in range(num_prototypes):
                prototype_img_path = f"{prototype_img_folder}/prototype-{idx}.png"
                prototype_img = plt.imread(prototype_img_path)
                plt.imshow(prototype_img, cmap='gray')
                plt.title(f'Prototype {idx}')
                plt.show()