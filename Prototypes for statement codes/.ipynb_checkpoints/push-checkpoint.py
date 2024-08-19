import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

from preprocess import normalize_img

from settings import num_scp_codes

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)      
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
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
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(torch.tensor(prototype_update, dtype=torch.float32).cuda())
    # prototype_network_parallel.cuda()
    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))

# update each prototype for current search batch
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
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        # print('preprocessing input for pushing ...')
        # search_batch = copy.deepcopy(search_batch_input)
        search_batch = preprocess_input_function(search_batch_input)

    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy()) #(batch_size, depth, h, w)
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy()) #(batch_size, num_protos, h, w)

    del protoL_input_torch, proto_dist_torch

    if class_specific:

        # ------------------------
        # class_to_img_index_dict = {key: [] for key in range(num_classes)}
        # for img_index, img_y in enumerate(search_y):
        #     img_label = img_y.item()
        #     class_to_img_index_dict[img_label].append(img_index)
        # ------------------------
        # Create a scp_code_class to images in current batch lookup.
        scp_labels = search_y[1]  # batch_size x num_prototypes/num_scp_codes
        class_to_img_index_dict = {key: [] for key in range(num_scp_codes)}
        # Get indices of the scp codes which are activated for each example
        non_zero_indices = [list(row.nonzero(as_tuple=True)[0].tolist()) for row in scp_labels]
        for example_number, row in enumerate(non_zero_indices):
            for activated_scp_class in row:
                class_to_img_index_dict[activated_scp_class] += [example_number]
        # ------------------------

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        #if n_prototypes_per_class != None:
        if class_specific:
            # target_class is the class of the class_specific prototype
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            # if there is not images of the target_class from this batch
            # we go on to the next prototype
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            # shape of proto_dist_ = (batch_size, num_prototypes, h, w)
            # shape of proto_dist_j = (examples_of_particular_class, h, w)
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            # if it is not class specific, then we will search through
            # every example
            # shape of proto_dist_ = (batch_size, h, w)
            proto_dist_j = proto_dist_[:,j,:,:]

        # minimum value in the proto_dist_j
        batch_min_proto_dist_j = np.amin(proto_dist_j)
        # if the minimum value is less than the global minimum, find its coordinates
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
                # shape of batch_argmin_proto_dist_j = [3],i.e., [image_index_fake, h_index, w_index]
            if class_specific:
                '''
                change the argmin index from the index among
                images of the target class to the index in the entire search
                batch
                '''
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            # ----------------------------
            # retrieve the corresponding feature map patch (rf of protoL on latent space or feature map)
            # ----------------------------
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :, # channels
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]
 
            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # ----------------------------
            # get the receptive field boundary of the image patch
            # that generates the representation (rf of protoL on the actual image)
            # ----------------------------
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            # get the whole image
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            
            # crop out the receptive field
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            # CHANGE THE PROTOTYPE TO BE THE WHOLE IMAGE 
            proto_img_j = original_img_j
       

            # ----------------------------
            # save the prototype
            # ----------------------------
            if dir_for_saving_prototypes is not None:
                if prototype_img_filename_prefix is not None:
                    # save the prototype image (highly activated region of the whole image)
                    proto_img_j = normalize_img(proto_img_j)
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + str(j) + '.png'),
                               proto_img_j,
                               vmin=0.0,
                               vmax=1.0)
                
    if class_specific:
        del class_to_img_index_dict
