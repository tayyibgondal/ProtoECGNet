train_batch_size = 32
test_batch_size = 32
input_size = 71 # Size of the SCP encoded vector
num_classes = 5 # Number of classes for diagnostic labels
lr = 1e-4
num_epochs = 20

train_df = 'train-100-scptolabels.json'
test_df = 'test-100-scptolabels.json'