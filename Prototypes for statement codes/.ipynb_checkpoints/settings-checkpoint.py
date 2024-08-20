# TODO: INCREASE THE DENSE HEAD AT THE END OF MODEL TO SEE IF IT WORKS WELL...

# base_architecture_to_features = {'resnet18': resnet18_features,
#                                  'resnet34': resnet34_features,
#                                  'resnet50': resnet50_features,
#                                  'resnet101': resnet101_features,
#                                  'resnet152': resnet152_features,
#                                  'densenet121': densenet121_features,
#                                  'densenet161': densenet161_features,
#                                  'densenet169': densenet169_features,
#                                  'densenet201': densenet201_features,
#                                  'vgg11': vgg11_features,
#                                  'vgg11_bn': vgg11_bn_features,
#                                  'vgg13': vgg13_features,
#                                  'vgg13_bn': vgg13_bn_features,
#                                  'vgg16': vgg16_features,
#                                  'vgg16_bn': vgg16_bn_features,
#                                  'vgg19': vgg19_features,
#                                  'vgg19_bn': vgg19_bn_features}

# DISCLAIMER: The following two are not used from settings.py, SPECIFY VIA TERMINAL BEFORE RUNNING THE MAIN.PY
base_architecture = 'resnet18'  # Choose one of Keys from above dictionary
experiment_run = '001'

bootstraping = True

img_size = 500  # MANIPULATE ?
num_classes = 5
num_scp_codes = 71
num_prototypes_for_each_class = 32  # MANIPULATE ?
num_prototypes = num_scp_codes * num_prototypes_for_each_class
prototype_shape = (num_prototypes, 256, 1, 1)  # MANIPULATE ?
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

# ----------------------------------------
# Following two settings are specifically for classification of 5 super classes
train_information = 'train-100.json'
test_information = 'test-100.json'

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

push_start = 10
push_epochs = [i for i in range(num_train_epochs) if i % 10 == 0]

# for saving a trained model checkpoint
target_auroc = 0.90