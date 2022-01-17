README.txt
--------------------------------------------------------------------------------
Bittensor Technical Challenge

Stephen Mosher (2022)

Short Description:
------------------
This repo represents my attempt at the Bittensor Technical Challenge I was 
given. The challenge involved pruning a trained neural network, up to some 
specified amount, and then evaluating its performance on a test set after the
pruning operation. A more thorough analysis of my results can be found on this
repo in the PDF titled "ANALYSIS".

I attempted this challenge two different ways. First, I tackled the challenge
using a TensorFlow implementation, which is what I'm more familiar with. Second,
I tackled the challenge using a PyTorch implementation. No doubt, the PyTorch 
implementation is a little bit rougher than the TensorFlow implementation, since
PyTorch is less familiar to me, but I'm happy with how both approaches turned 
out, at least for the purpose of this challenge. Undoubtedly, working code can
always be refined.

It should be noted that the code contained in this repo is very bare-bones, in 
the sense that I tried to keep things as simple as possible. For example, when
tackling a machine learning problem, a lot of care must be given to data pre-
processing, however, in the interest of expediency I skipped any and all data 
pre-processing. Likewise, I didn't really take any steps to evaluate/modify the
training process while underway. In both implementations, I just set the 
networks to train for 20 epochs and took the final result, no questions asked. 
In TensorFlow I know how to implement checkpoints, callbacks, early-stopping, 
learning rate reductions etc., and I'm sure this can be done in PyTorch as well.
But, again, I neglected any of that stuff so that I could complete this 
challenge in a reasonable amount of time. I also opted not to use any GPUs and
create something that could run on my local machine.

Structure:
----------
Each implementation follows the same organizational structure. All code related
to the TensorFlow implementation can be found in the "TensorFlow" directory, and
all code related to the PyTorch implementation can be found in the "PyTorch" 
directory. 

Each directory contains 2 files, a main script, called "BT_challenge…" and a
secondary file containing all utility functions, called "utils_...". In each 
case, these files are suffixed with either "_TF.py" or "PT.py", indicating the
implementation.

To Run:
--------
To run either implementation, simply access the top-level directory 
corresponding to the implementation of interest, then run the "BT_challenge…" 
script. Upon execution, these scripts will create folders to contain all the 
output they generate. Each implementation will generate two model files, and two
figures. The first figure plots simple learning curves of the training process 
(solid lines indicate the training data, dashed indicated the validation data).
The second figure displays the results of the network's performance against a 
test data set as a function of the degree to which it has been pruned. 

Requirements:
-------------
To run the scripts, you'll need the following packages

- Python 3
- Numpy
- Matplotlib
- TensorFlow
- PyTorch