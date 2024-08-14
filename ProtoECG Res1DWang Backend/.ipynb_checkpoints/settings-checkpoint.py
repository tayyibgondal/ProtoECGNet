experiment_run = '001'

img_size = 224  # Not used...
num_classes = 5
num_prototypes_for_each_class = 512  # MANIPULATE ?
num_prototypes = num_classes * num_prototypes_for_each_class
prototype_shape = (num_prototypes, 128, 1, 1)  # MANIPULATE ?
prototype_activation_function = 'log'
add_on_layers_type = 'regular'  # Currently code only works with this setting


data_path = ''  #TODO: Not USED, REMOVE IMPORTS FROM FILES
test_dir = data_path + 'test/' #TODO: Not USED, REMOVE IMPORTS FROM FILES
train_push_dir = data_path + 'train/' #TODO: Not USED, REMOVE IMPORTS FROM FILES

# ----------------------------------------
# Following two settings are specifically for classification of 5 super classes
freq = 100
train_information = f'train-{freq}.csv'
test_information = f'test-{freq}.csv'

train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

# Specify the number of examples to load (set them to None to load all examples)
num_train_examples = None  
num_test_examples = None 

# Logging directory for results
log_dir = 'logs'

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 1000 
num_warm_epochs = 5

push_start = 15
push_epochs = [i for i in range(num_train_epochs) if i % push_start == 16]

# useful in inference file
# # For two classes only
# label_index_to_label_text_mapping = {
#     0: 'Normal',
#     1: 'Abnormal'
# }
# For 5 classes
label_index_to_label_text_mapping = {'CD': 0, 'HYP': 1, 'MI': 2, 'NORM': 3, 'STTC': 4}

# for saving a trained model checkpoint
target_auroc = 0.70
target_f1 = 0.70

# LEARNINGS
# Increasing cross entropy coefficient won't help.