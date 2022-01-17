'''
FUNCTION SET utils_TF.py

This file contains a set of functions that get called when running my
TensorFlow implementation of the Bittensor challenge. Descriptions for each 
function are below.

Stephen M. January, 2022
'''

#################################### IMPORTS ###################################

# Imports required for the functions in this script.
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Sequential

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
# Returns 4 numpy arrays: the MNIST images for training and testing, and the
# corresponding labels. The arrays containing the MNIST images are flattened,
# and the arrays containing the label data are converted to categorical data. 
def grab_and_prep_MNIST():

  # Use TensorFlow's built-in module to grab MNIST, conveniently splits into
  # train and validation/test sets.
  (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

  # Determine the number of training and test examples.
  N_train_examples = X_train.shape[0]
  N_test_examples = X_test.shape[0]

  # Determine the pixel dimensions of the MNIST images.
  image_dims = X_test.shape[1::]

  # Calculate what the flattened length of the images will be.
  image_len_flat = image_dims[0] * image_dims[1]

  # Re-shape/flatten MNIST image data. Not using a convolutional neural net.
  X_train = X_train.reshape(N_train_examples, image_len_flat)
  X_test = X_test.reshape(N_test_examples, image_len_flat)

  # B/c this is a 10-class classification problem, we need to modify the MNIST
  # label data. Each example needs to be an array of length 10, with zeros 
  # everywhere, except for a single 1 in the position corresponding to the
  # label. (i.e. an MNIST image with label 2 is mapped to [0,0,1,0,0,0,0,0,0,0])
  # TensorFlow has a convenient built-in function for this purpose. 
  Y_train = to_categorical(Y_train)
  Y_test = to_categorical(Y_test)

  # Return
  return (X_train, Y_train, X_test, Y_test)

################################################################################

# Constructs and compiles a simple neural network according to the architecture
# outlined in the Bittensor technical challenge.
def bittensor_neural_net():

  # Use TensorFlow's Sequential module to construct a simple neural net
  NN = Sequential([
            Dense(1000, input_shape=(784,), activation='relu', use_bias=False),
            Dense(1000, activation='relu', use_bias=False),
            Dense(500, activation='relu', use_bias=False),
            Dense(200, activation='relu', use_bias=False),
            Dense(10, activation='softmax', use_bias=False)])

  # Print out a summary of the network.
  NN.summary()

  # Compile the network with an ADAM optimizer and categorical cross-entropy
  # loss function. For metrics, keep track of accuracy. 
  NN.compile(optimizer=Adam(), 
             loss='categorical_crossentropy',
            metrics=['accuracy'])

  return NN

################################################################################

# Generate some plots of the training and validation process.
def plot_training(history):
  print()
  print('Ploting results of training process...')
  print()
 
  fig, ax1 = plt.subplots()
  ax1.set_xlabel('Epoch')
  
  ax1.plot(history.history['loss'], 'b', lw=1.5, label='Training')
  ax1.plot(history.history['val_loss'], 'b', lw=1.5, ls='--', label='Testing')
  ax1.set_ylabel('Loss', color='b')
  ax1.tick_params(axis='y', labelcolor='b')
  ax1.legend()
  
  ax2 = ax1.twinx()
  
  ax2.plot(np.array(history.history['accuracy']) * 100, 'r', lw=1.5)
  ax2.plot(np.array(history.history['val_accuracy']) * 100, 'r', lw=1.5, ls='--')
  ax2.set_ylabel('Accuracy [%]', color='r')
  ax2.tick_params(axis='y', labelcolor='r')
  
  fig.tight_layout()
  
  plt.savefig('../figures/NN_TF_training.png')
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
  plt.savefig('../figures/TF_sparsity_v_accuracy.png')
  plt.clf()
  plt.close()

################################################################################

# Perform weight-pruning on a trained neural network and test its performance.
def weight_pruning(k_percent, X_test, Y_test):

  # Numpy array to hold the network performance results after each pruning.
  performance_scores = np.zeros(len(k_percent))

  for i, k in enumerate(k_percent):
    # Load trained network. Want to reload fresh copy of trained network for each
    # time we prune.
    network = load_model('../models/NN_TF.h5')

    # Loop over layer weights in the network except for those in the final layer.
    for j, layer in enumerate(network.layers[0:-1]):

      # Handy print statement.
      print('Weight pruning layer ' + str(j + 1) + '/5, for k=' +str(k*100) +' %')
  
      # Grab weights for current layer.
      layer_weights = layer.get_weights()[0]
  
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
      layer.set_weights([layer_weights])

    print()

    # After looping over all the layers and weight-pruning, evaluate the 
    # performance of the pruned network on the test set.
    N_test_examples = X_test.shape[0]
    results = np.argmax(network.predict(X_test), axis=1)
    score = np.sum(results == np.argmax(Y_test, axis=1)) / N_test_examples

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
def unit_pruning(k_percent, X_test, Y_test):

  # Numpy array to hold the network performance results after each pruning.
  performance_scores = np.zeros(len(k_percent))

  for i, k in enumerate(k_percent):
    # Load trained network. Want to reload fresh copy of trained network for 
    # each time we prune.
    network = load_model('../models/NN_TF.h5')

    # Loop over layer weights in the network except for those in the final layer
    for j, layer in enumerate(network.layers[0:-1]):

      # Handy print statement.
      print('Unit-pruning layer ' + str(j + 1) + '/5, for k=' +str(k*100) +' %')
  
      # Grab weights for current layer.
      layer_weights = layer.get_weights()[0]

      # Count number of weight columns in current layer.
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
      layer.set_weights([layer_weights])
    
    print()

    # After looping over all the layers and unit-pruning, evaluate the 
    # performance of the pruned network on the test set.
    N_test_examples = X_test.shape[0]
    results = np.argmax(network.predict(X_test), axis=1)
    score = np.sum(results == np.argmax(Y_test, axis=1)) / N_test_examples

    # Store current score
    performance_scores[i] = score

  # Return performance scores
  return performance_scores