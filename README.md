# cs6910_assignment_1

i) NN_model.py : Python script that contains the definition of the "nn_model" class whose objects are the feed-forward neural network model used in all
other python scripts in the assignment, by importing this class definition in other scripts.

- Important class attributes of "nn_model" :  Weights(W),bias(b),layer pre-activation values(a), layer activated values(h), their coressponding gradients w.r.t to loss grad_W,grad_b,grad_a,grad_h. All of these are dictionary instead of numpy objects to support different hidden layer sizes for each layer. Each element in the dictionaries are numpy objects corresponding to each layer.

- Important class methods of "nn_model" :
  - compact_build() : to initialise and build the neural network of desired configuration
  - training(): does splitting,shuffling,augmentation with help of helper methods and calls the optimizer functions/methods corespponding to the input/arguments passed     to training.
  
  ii) train.py : Python script to train nn_models and return/print the test results of the model whose parameters can be passed from the command line as specified in the code specifications. 
  - Arguments are parsed from the terminal using "argparse" module
  - running "python -train.py -h" will bring up the help page with details regarding each argument, their usage, their default value.
  
  iii) data_images_and_sweeps.py: Python script with code for the logging images of fashion mnist and the 3 sweep experiments done.
  
  iv) best_config_confusion_matrix.py: Python script that trains and logs the best configuration model to wandb. Also,plots and logs the resulting confusion matrix for fashion-mnist test datset to my wandb project.
  
  v) mnist_test.py: Python script that trains the three recommended network configurations over mnist datset and prints the test accuracies for each model. No training runs sent to my wandb project,only printing of training,validation accuracies and losses.
  
  
