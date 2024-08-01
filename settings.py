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

base_architecture = 'resnet18'  # Choose one of Keys from above dictionary
img_size = 224
prototype_shape = (60, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = '001'

data_path = ''  #TODO: Not USED, REMOVE IMPORTS FROM FILES
test_dir = data_path + 'test/' #TODO: Not USED, REMOVE IMPORTS FROM FILES
train_push_dir = data_path + 'train/' #TODO: Not USED, REMOVE IMPORTS FROM FILES

# ----------------------------------------
# Path to the CSV file
csv_file_for_labels = '../../../data/padmalab_external/special_project/physionet.org/files/ptb-xl/1.0.3/ptbxl_train_label_df.csv'
# Path to the image directory
data_dir = '../../../data/padmalab_external/special_project/physionet.org/files/ptb-xl/1.0.3/records100_ground_truth'
train_dir = data_path + '../../../data/padmalab_external/special_project/physionet.org/files/ptb-xl/1.0.3/records100_ground_truth'
train_batch_size = 80
test_batch_size = 100
train_push_batch_size = 75

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
