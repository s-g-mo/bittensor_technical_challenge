'''
SCRIPT BT_challenge_PT.py

This script is my PyTorch implementation of the Bittensor technical challenge.
It calls several utility functions (contained in the utils_PT.py file) while
executing. The script does the following:

  1) Loads and prepares MNIST image data such that it can be passed into a 
  simple neural network.

  2) Builds and trains the network.

  3) Then examines the effects of purning the network according to two different
  techniques.

Trained models and figures generated by this script are saved in '../models/'
and '../figures/' directories, which are created upon execution.

Stephen M. January, 2022
'''

#################################### IMPORTS ###################################

# Standard imports
import torch
import numpy as np
from torchvision import datasets, transforms

# Imports the utility functions written for this challenge.
import utils_PT

##################################### MAIN #####################################

# Setup directories to store trained neural networks and figures.
utils_PT.directory_setup()

# Grab and prepare MNIST data so that it can be passed into the network.
train_loader, test_loader = utils_PT.grab_and_prep_MNIST()

# Build and compile the network (architecture hard-coded as per the challenge).
network = utils_PT.bittensor_neural_net()

# Train/validate the network for 20 epochs. Initialize empty lists to store avg
# training/validation metrics after each epoch.
epochs = 20
avg_train_losses = []
avg_train_accs = []
avg_test_losses = []
avg_test_accs = []

print('Training...')
for epoch in range(epochs):

  # Training loop.
  train_metrics = utils_PT.train_loop(train_loader, network, epoch)
  
  # Store train metrics of current epoch.
  avg_train_losses.append(train_metrics[0])
  avg_train_accs.append(train_metrics[1])

  # Validation loop.
  test_metrics = utils_PT.validation_loop(test_loader, network)

  # Store validation metrics of current epoch.
  avg_test_losses.append(test_metrics[0])
  avg_test_accs.append(test_metrics[1])

# Save the network to the 'models' directory. Save two copies, one as a backup
# in case I want to play around with the network later.
torch.save(network, '../models/NN_PT.pt')
torch.save(network, '../models/NN_PT_backup.pt')

# Plot the results of the training process. Save to '../figures/' dir.
utils_PT.plot_training(avg_train_losses, 
                       avg_test_losses,
                       avg_train_accs, 
                       avg_test_accs)

# Numpy array of k-th percents to loop over for pruning.
k_percents = np.array([0, 25, 50, 60, 70, 80, 90, 95, 97, 99])/100

# Load test data so we can pass into the pruning functions to evaluate perf.
transform = transforms.ToTensor()
test = datasets.MNIST('../data', train=False, transform=transform)

# Prune trained neural network using weight-pruning and unit-pruning, then
# measure network performance against the test data.
weight_pruning_scores = utils_PT.weight_pruning(k_percents, test)
unit_pruning_scores = utils_PT.unit_pruning(k_percents, test)

# Plot network performance vs pruning.
utils_PT.plot_pruning_performance(k_percents,
                                  weight_pruning_scores, 
                                  unit_pruning_scores)