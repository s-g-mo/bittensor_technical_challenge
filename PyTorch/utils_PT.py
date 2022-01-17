'''
FUNCTION SET utils_PT.py

This file contains a set of functions thatt get called when running my
TensorFlow implementation of the Bittensor challenge. Descriptions for each 
function are below.

Stephen M. January, 2022
'''

#################################### IMPORTS ###################################

# Imports required for the functions in this script.
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

################################### FUNCTIONS ##################################

# Simple function to set up directories to store trained networks and figures
# created for this challenge.
def directory_setup():
  models_dir = '../models/'
  figures_dir = '../figures/'

  for directory in [models_dir, figures_dir]:
    if not os.path.isdir(directory):
      print()
      print('Directory ' + directory + ' doesn`t exist - creating it.')
      print()
      os.makedirs(directory)

################################################################################

# Grabs MNIST data and prepares it to be passed into a simple neural network.
# Returns 2 PyTorch DataLoader objects. One contains the MNIST images and labels
# for the training set, the other, those for the testing set.
def grab_and_prep_MNIST():

  # Set batch size for training and testing loops. Can be different in 
  # principle, will keep the same just to simplify things.
  batch_size = 100
  
  # Declare a transforms object in order to transform MNIST data to PyTorch
  # tensors upon loading.
  transform = transforms.ToTensor()

  # Grab MNIST data using PyTorch's built-in capabilities.
  train = datasets.MNIST('../data', download=True, train=True, transform=transform)
  test = datasets.MNIST('../data', download=True, train=False, transform=transform)
  
  # Populate PyTorch DataLoader objects with the train/test data.
  train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

  # Return data loaders.
  return(train_loader, test_loader)

################################################################################

# Constructs and compiles a simple neural network according to the architecture
# outlined in the Bittensor technical challenge.
def bittensor_neural_net():

  # Use PyTorch's Sequential module to construct a simple neural net
  NN = nn.Sequential(nn.Linear(784, 1000, bias=False), nn.ReLU(), 
                     nn.Linear(1000, 1000, bias=False), nn.ReLU(),
                     nn.Linear(1000, 500, bias=False), nn.ReLU(),
                     nn.Linear(500, 200, bias=False), nn.ReLU(),
                     nn.Linear(200, 10, bias=False), nn.LogSoftmax(dim=1))
  
  # Print out a summary of the network.
  print()
  print('Network Summary')
  print('---------------')
  print()
  print(NN)
  print()

  return NN

################################################################################

# Training loop.
def train_loop(train_loader, network, epoch):

  # Initialize empty lists to store training metrics.
  train_losses = []
  train_accs = []

  # Use negative log least likelihood loss function and ADAM optimizer.
  loss_func = torch.nn.NLLLoss()
  optimizer = torch.optim.Adam(network.parameters())

  # Place the network in train mode (just to be safe).
  network.train()

  # Loop over batches of training data.
  for batch_idx, (X_train, Y_train) in enumerate(train_loader):

    # Determine batch size from train data batch.
    batch_size = X_train.shape[0]

    # Set gradients to zero each forward pass.
    optimizer.zero_grad()
    
    # Compute network outputs of current batch. Flatten input.
    output = network(X_train.view(batch_size, -1))
    predicted = torch.max(output.data, axis=1)[1]
    
    # Count number of correct predictions.
    correct = (predicted == Y_train).sum()

    # Compute loss.
    loss = loss_func(output, Y_train)

    # Network update step.
    loss.backward()
    optimizer.step()

    # Print statement (every 10 batches).
    if batch_idx % 10 == 0:
      print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(X_train), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

    # Append training loss and accuracy of current batch to master lists.
    train_losses.append(loss.item())
    train_accs.append((correct/batch_size).item())

  # Once training is complete for the epoch, compute average loss and acc.
  avg_train_losses = np.mean(train_losses)
  avg_train_accs = np.mean(train_accs)

  # Return the metrics.
  return [avg_train_losses, avg_train_accs]

################################################################################

# Validation loop
def validation_loop(test_loader, network):

  # Place network in validation mode.
  network.eval()

  # Initialize metrics.
  loss = 0
  correct = 0

  # Use negative log least likelihood loss function.
  loss_func = torch.nn.NLLLoss() 

  # Don't update weights/gradients during validation.
  with torch.no_grad():

    # Loop over batches of test data.
    for X_test, Y_test in test_loader:

      # Determine batch size from train data batch.
      batch_size = X_test.shape[0]

      # Compute network outputs of current batch. Flatten input.
      output = network(X_test.view(batch_size, -1))
      predicted = torch.max(output.data, axis=1)[1]

      # Add current loss to running tally of total loss.
      loss += loss_func(output, Y_test)
      
      # Count number of correct predictions.
      correct += (predicted == Y_test).sum()

    # Once testing is complete for the epoch, compute average loss and acc. 
    avg_test_loss = np.round(loss.numpy()/batch_size, 3)
    avg_test_acc = (correct/len(test_loader.dataset)).numpy()

    # Print out average test metrics for current epoch.
    print()
    print('Average Test Loss: '+str(avg_test_loss)+'    Average Test Accuracy: '
          +str(np.round(avg_test_acc*100,4))+'%')
    print()
    
    # Return the metrics.
    return [avg_test_loss, avg_test_acc]

################################################################################

# Generate some plots of the training and validation process.
def plot_training(m1, m2, m3, m4):
  print()
  print('Ploting results of training process...')
  print()
 
  fig, ax1 = plt.subplots()
  ax1.set_xlabel('Epoch')
  
  ax1.plot(m1, 'b', lw=1.5, label='Training')
  ax1.plot(m2, 'b', lw=1.5, ls='--', label='Testing')
  ax1.set_ylabel('Loss', color='b')
  ax1.tick_params(axis='y', labelcolor='b')
  ax1.legend()
  
  ax2 = ax1.twinx()
  
  ax2.plot(np.array(m3) * 100, 'r', lw=1.5)
  ax2.plot(np.array(m4) * 100, 'r', lw=1.5, ls='--')
  ax2.set_ylabel('Accuracy [%]', color='r')
  ax2.tick_params(axis='y', labelcolor='r')
  
  fig.tight_layout()
  
  plt.savefig('../figures/NN_PT_training.png')
  plt.clf()
  plt.close()

################################################################################

# Plot performance of network as a function of k% pruning for both methods.
def plot_pruning_performance(k_percent, wp_scores, up_scores):
  print()
  print('Ploting network performance vs. pruning...')
  print()
  plt.plot(k_percent * 100, wp_scores * 100, alpha=0.8)
  plt.plot(k_percent * 100, up_scores * 100, alpha=0.8)
  plt.xlabel('Sparsity [%]')
  plt.ylabel('Accuracy [%]')
  plt.legend(['Weight Pruning', 'Unit/Neuron Pruning'], loc='lower left')
  plt.title('Bittensor Accuracy vs. Sparsity Challenge')
  plt.savefig('../figures/PT_sparsity_v_accuracy.png')
  plt.clf()
  plt.close()

################################################################################

# Perform weight-pruning on a trained neural network and test its performance.
def weight_pruning(k_percent, test):

  # Numpy array to hold the network performance results after each pruning.
  performance_scores = np.zeros(len(k_percent))

  for i, k in enumerate(k_percent):
    # Load trained network. Want to reload fresh copy of trained network for 
    # each time we prune.
    network = torch.load('../models/NN_PT.pt')

    # Loop over layer weights in the network except for those in the final layer
    for j, layer in enumerate(network.parameters()):
      if j == 4:
        continue

      # Handy print statement.
      print('Weight pruning layer ' + str(j + 1) + 
            '/5, for k=' +str(k*100) +' %')
      
      # Grab weights for current layer.
      layer_weights = layer.data.numpy()

      # Count the number of weights in current layer.
      num_layer_weights = layer_weights.size

      # Sort the weights in order of absolute magnitude.
      sorted_abs_layer_weights = np.sort(np.abs(layer_weights).flatten())
    
      # Determine the idx that separates the k-th% of weights from the rest.
      cutoff_idx = int(k * num_layer_weights)

      # Determine the value of the weight that separates the k-th% from the rest.
      cutoff_value = sorted_abs_layer_weights[cutoff_idx]

      # Zero out all layer weights less than the cuttoff value.
      layer_weights[np.abs(layer_weights) < cutoff_value] = 0
    
      # Update the weights after pruning.
      with torch.no_grad():
        layer = torch.tensor(layer_weights)

    print()

    # After looping over all the layers and weight-pruning, evaluate the 
    # performance of the pruned network on the test set.
    N_test_examples = test.data.shape[0]
    results = torch.argmax(network(test.data.view(N_test_examples, -1).type(torch.float)), axis=1)
    score = (results == test.targets).sum() / N_test_examples
    
    # Store current score
    performance_scores[i] = score

  # Return performance scores
  return performance_scores

################################################################################

# Simple function to compute the column-wise L2-norm of a 2D numpy array.
# Returns a 1D array of the L2-norms of each column.
def columnwise_L2_norm(matrix):
  return np.sqrt(np.sum(matrix**2, axis=0))

################################################################################

# Perform unit-pruning on a trained neural network and test its performance.
def unit_pruning(k_percent, test):

  # Numpy array to hold the network performance results after each purining.
  performance_scores = np.zeros(len(k_percent))

  for i, k in enumerate(k_percent):
    # Load trained network. Want to reload fresh copy of trained network for 
    # each time we prune.
    network = torch.load('../models/NN_PT.pt')

    # Loop over layer weights in the network except for those in the final layer
    for j, layer in enumerate(network.parameters()):
      if j == 4:
        continue

      # Handy print statement.
      print('Unit-pruning layer ' + str(j + 1) + '/5, for k=' +str(k*100) +' %')
      
      # Grab weights for current layer.
      layer_weights = layer.data.numpy()

      # Count number of columns of layer weights.
      num_cols = layer_weights.shape[1]

      # Compute the L2-norm on each column.
      column_L2s = columnwise_L2_norm(layer_weights)
      
      # Sort the L2-norms.
      sorted_column_L2s = np.sort(column_L2s.flatten())
  
      # Determine the idx that separates the k-th% of L2s from the rest.
      cutoff_idx = int(k * num_cols)

      # Determine the L2 value that separates the k-th% from the rest.
      cutoff_value = sorted_column_L2s[cutoff_idx]

      # Zero out all weight columns whose L2 is less than the cuttoff value.
      layer_weights[:, (column_L2s < cutoff_value) == True] = 0
  
      # Update the weights after pruning.
      with torch.no_grad():
        layer = torch.tensor(layer_weights)
  
    print()

    # After looping over all the layers and weight-pruning, evaluate the 
    # performance of the pruned network on the test set.
    N_test_examples = test.data.shape[0]
    results = torch.argmax(network(test.data.view(N_test_examples, -1).type(torch.float)), axis=1)
    score = (results == test.targets).sum() / N_test_examples

    # Store current score
    performance_scores[i] = score

  # Return performance scores
  return performance_scores