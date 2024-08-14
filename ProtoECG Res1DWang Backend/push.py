import torch
import numpy as np
import os
import time

def makedir(dir_path):
    """Create a directory if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_dat_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype waveform comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    global_min_proto_dist = np.full(n_prototypes, np.inf)      
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2]])

    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 4], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 4], fill_value=-1)

    if root_dir_for_saving_prototypes is not None:
        if epoch_number is not None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes, 'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_dat_filename_prefix=prototype_dat_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir is not None and proto_bound_boxes_filename_prefix is not None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

# update each prototype for the current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_dat_filename_prefix='prototype',
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    # if preprocess_input_function is not None:
    #     search_batch_input = preprocess_input_function(search_batch_input)

    with torch.no_grad():
        search_batch = search_batch_input.cuda()
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    max_dist = prototype_shape[1] * prototype_shape[2]

    for j in range(n_prototypes):
        if class_specific:
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:, j, :]
        else:
            proto_dist_j = proto_dist_[:, j, :]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape))
            if class_specific:
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_end_index = fmap_start_index + prototype_shape[2]

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch, :, fmap_start_index:fmap_end_index]

            # Save the full waveform instead of just the patch
            full_waveform = search_batch_input[img_index_in_batch].cpu().numpy()
            if dir_for_saving_prototypes is not None and prototype_dat_filename_prefix is not None:
                prototype_filename = os.path.join(dir_for_saving_prototypes, f"{prototype_dat_filename_prefix}{j}.dat")
                np.save(prototype_filename, full_waveform)

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = np.squeeze(batch_min_fmap_patch_j, axis=2)

            proto_rf_boxes[j, 0] = img_index_in_batch + start_index_of_search_batch
            proto_rf_boxes[j, 1] = fmap_start_index
            proto_rf_boxes[j, 2] = fmap_end_index
            if proto_rf_boxes.shape[1] == 5 and search_y is not None:
                proto_rf_boxes[j, 3] = search_y[img_index_in_batch].item()

