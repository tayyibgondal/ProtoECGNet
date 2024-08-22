base_architecture = 'resnet18'  # Choose one of Keys from above dictionary
experiment_run = '001'

bootstraping = True

img_size = 224  # MANIPULATE ?
num_classes = 5
num_scp_codes = 71
num_prototypes_for_each_class = 32  # MANIPULATE ?
num_prototypes = num_scp_codes * num_prototypes_for_each_class
prototype_shape = (num_prototypes, 128, 1, 1)  # MANIPULATE ?
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

# ----------------------------------------
# Following two settings are specifically for classification of 5 super classes
train_information = 'train-100.json'
test_information = 'val-100.json'
mappings_file = 'label_mappings.txt'

train_batch_size = 100
test_batch_size = 80
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

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

# for saving a trained model checkpoint
target_auroc = 0.80