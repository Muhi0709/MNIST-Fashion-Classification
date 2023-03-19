import argparse
parser=argparse.ArgumentParser(prog="Training & Testing NN model",description="""Training the feed forward neural network with the best c
onfiguration on validation set obtained from WandB Sweep experiments.All parse arguments have default values and are set to the best configuration""")
parser.add_argument('-wp','--wandb_project',default="cs6910_assignment_1",
                    help="Project name used to track experiments.Default set to 'cs6910_assignment_1'.")
parser.add_argument('-we','--wandb_entity',default="ep19b005",
                    help="Wandb entity used to track experiments in the W&B dashboard.Default set to 'ep19b005'.")
parser.add_argument('-d','--dataset',default="fashion_mnist",choices=["fashion_mnist","mnist"],
                    help="Choices of dataset to train over.Supported choices are 'fashion_mnist','mnist'.Default set to 'fashion_mnist'")
parser.add_argument('-e','--epochs',default=15,type=int,help="Number of epochs to train the nueral network.Supported type 'int'.Default set to 15")
parser.add_argument('-b','--batch_size',default=64,type=int,
                    help="Batch size used to train neural network.Supported type 'int'.Default set to 64")
parser.add_argument('-l','--loss',default="cross_entropy",choices=["cross_entropy","mse","mean_squared_error"],
                    help="""Loss function used to train the neural network.Supported choices are 'cross_entropy','mse','mean_squared_error'.
For mean squared error loss,both 'mse' & 'mean_squared_error' can be used.Default set to 'cross_entropy'""")
parser.add_argument('-o','--optimizer',default="nadam",choices=["GD","SGD","sgd","momentumGD","momentum","Nesterov","nag","Adagrad","Mini_batch_GD",
                     "RMSprop","rmsprop","Adam","adam","NAdam","nadam"],help="""Optimizer used for training.Supported choices are 'GD','SGD','sgd','momentumGD','momentum','Nesterov','nag',
'Adagrad','Mini_batch_GD','RMSprop','rmsprop','Adam','adam','NAdam','nadam'.
For SGD,use 'sgd' or 'SGD'.For Nesterov optimizer,use 'nag' or 'Nesterov'.For momentum optimizer use 'momentumGD' or 'momentum'.
For RMSprop optimizer,use 'RMSprop' or 'rmsprop'.For ADAM optimizer,use 'Adam' or 'adam'.For Nadam optimizer, use 'NAdam' or 'nadam'.
Default set to 'nadam'""")
parser.add_argument('-lr','--learning_rate',default=0.00025,type=float,help="Learning rate used to optimize model parameters.Supported type 'float'.Default set to 0.00025")
parser.add_argument('-m','--momentum',default=0.9,type=float,help="Momentum used by momentum and nag optimizers.Supported type 'float'.Default set to 0.9")
parser.add_argument('-beta','--beta',default=0.9,type=float,help="Beta used by rmsprop optimizer.Supported type 'float'.Default set to 0.9")
parser.add_argument('-beta1','--beta1',default=0.99,type=float,help="Beta1 used by adam and nadam optimizer.Supported type 'float'.Default set to 0.99")
parser.add_argument('-beta2','--beta2',default=0.999,type=float,help="Beta2 used by adam and nadam optimizer.Supported type 'float'.Default set to 0.999")
parser.add_argument('-eps','--epsilon',default=10**-8,type=float,help="Epsilon used by optimizers.Supported type 'float'.Default set to 10**-8")
parser.add_argument('-w_d','--weight_decay',default=0.001,type=float,help="L2 regularisation parameter or decay parameter used during training.Supported type 'float'.Default set to 0.001")
parser.add_argument('-w_i','--weight_init',default="Xavier",choices=["random","Xavier","He"],help=""" Weight and bias initialization used during instantiation.
Supported choices are 'random','Xavier' and 'He' initialisations.Default set to 'Xavier'.""" )
parser.add_argument('-nhl','--num_layers',default=5,type=int,help="Number of hidden layers in the feed forward neural network.Supported type 'int'.Default set to 5")
parser.add_argument('-sz','--hidden_size',default=64,type=int,help="Number of neurons in a hidden layer.Supported type 'int'.Default set to 64.")
parser.add_argument('-a','--activation',default="leakyreLU",choices=["identity","sigmoid","tanh","reLU","ReLU","leakyreLU"],help=""" 
Activation function for the neurons in the hidden layers.Supported choices are 'identity','sigmoid','tanh','reLU','ReLU',leakyreLU'.For relu activation, use 'reLU' or 'ReLU'
Default set to 'leakyreLU'.""")
parser.add_argument('-alpha','--leaky_alpha',type=float,default=0.1,help="""Non-trainable leaky alpha parameter used in training only if the 'activation' argument equals
"leakyreLU".Supported type 'float'.Default set to 0.1""")
parser.add_argument('-vp','--val_percent',type=float,default=0.1,help="""Percentage of training data to be split as validation data to be given as floating point number.
.Supported type 'float'.Default set to 0.1""")
parser.add_argument('-ncls','--num_of_classes',type=int,default=10,help="""Number of neurons in the output layer.Same as the number of classes in the data.
Supported type 'int'.Default set to 10 corresponding to fashion and mnist datasets.""")
parser.add_argument('-es','--early_stopping',dest="early_stopping",action="store_true",help="""Boolean flag (action='store_true') with a default False value.Do early stopping during training by tracking validation 
losses when set to True""")
parser.add_argument('-pat','--patience',default=2,type=int,help="""patience to be used while tracking validation loss for early stopping.Put to use only if 
'early_stopping argument' is set to True.Supported type 'int'.Default set to 2. """)
parser.add_argument('-dag','--data_augmentation',dest="data_augmentation",action="store_true",help="""Boolean flag (action='store_true') with a default False value.Do noise addition data augmentation to training data for training
when set to True""")
parser.add_argument('-seed','--seed',type=int,default=644,help="""seed value to be used by numpy random number generator during weight initialization,data shuffling,
batch shuffling,noise addition augmenatation to ensure reproducibility of result.Supported type 'int'.Default set to 644""")
parser.add_argument('-no_log','--no_wandb_log',dest="wandb_log",action='store_false',help=""" Boolean flag (action='store_false') with a default True value.It allows logging of training loss,training accuracy,
validation accuracy and validation loss (to a already initialized wandb run) during training of Neural Network. When called/flagged(becomes False), only printing of these values occur.
.Default set to True""" )

args=parser.parse_args()

from NN_model import nn_model
from data_loader import get_fashion_mnist,get_mnist

if args.wandb_log:
    import wandb
    wandb.login()
    configuration={
        "dataset": args.dataset,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "beta": args.beta,
        "beta1":args.beta1,
        "beta2":args.beta2,
        "epsilon":args.epsilon,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "weight_init": args.weight_init,
        "activation": args.activation,
        "loss": args.loss,
        "leakyalpha": args.leaky_alpha,
        "val_percent": args.val_percent,
        "num_of_classes": args.num_of_classes,
        "early_stopping": args.early_stopping,
        "patience": args.patience,
        "data_augmentation": args.data_augmentation,       
    }
    run=wandb.init(project=args.wandb_project,entity=args.wandb_entity,config=configuration)
    config=wandb.config
    name="ep_{}_bs_{}_hlnum_{}_hlsize_{}_lr_{}_opt_{}_init_{}_act_{}_loss_{}_l2_{}_leakyalpha_{}_early_{}_patie_{}_augment_{}".format(config.epochs,config.batch_size,
                                                                                                                        config.num_layers,
                                                                                                                        config.hidden_size,config.learning_rate,
                                                                                                                        config.optimizer,config.weight_init,
                                                                                                                        config.activation,config.loss,
                                                                                                                        config.weight_decay,config.leakyalpha,config.early_stopping,
                                                                                                                        config.patience,
                                                                                                                        config.data_augmentation)
    run.name=name

data=None
print("loading dataset")
if args.dataset=="fashion_mnist":
    data=get_fashion_mnist()
else:
    data=get_mnist()

print("dataset loaded successfully")

(X_train,y_train),(X_test,y_test)=data
input_dimension = X_train[0].shape[0]
num_of_classes=args.num_of_classes
num_of_neurons_list = [args.hidden_size]*(args.num_layers) +[num_of_classes]
activation_func_list = ([args.activation]*(args.num_layers) + ["softmax"]) if args.activation!="leakyreLU" else ([(args.activation,args.leaky_alpha)]*(args.num_layers) + ["softmax"])

model=nn_model()
model.compact_build_nn(input_dimension,args.num_layers+1,num_of_neurons_list,activation_func_list,
                       args.weight_init,args.seed)

print("Configuration used:")
print(args)
print("-"*200)

model.training(X_train,y_train,val_percent=args.val_percent,loss_func= args.loss,optimizer=args.optimizer,
               max_epoch=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate,
               beta=args.momentum,rmsbeta=args.beta,beta_1=args.beta1,beta_2=args.beta2,epsilon=args.epsilon,L2_decay=args.weight_decay,
               early_stopping=args.early_stopping,patience=args.patience,data_augmentation=args.data_augmentation,seed=args.seed,wandb_log=args.wandb_log)

y_test_encoded=model.one_hot_encoding(y_test)
acc,loss=model.compute_loss_accuracy(X_test,y_test_encoded,args.loss)
print("=>{} Test Accuracy:".format(args.dataset),acc)
print("=>{} Test loss:".format(args.dataset),loss)
print("-"*200)





