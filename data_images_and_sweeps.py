from keras.datasets import fashion_mnist
from NN_model import nn_model
import numpy as np
import wandb
wandb.login()
def load_fashion_mnist(scaling="MinMax"):
    r=np.random.default_rng(seed=100000)
    wandb.init(project="cs6910_assignment_1",name="data_examples",entity="ep19b005")
    
    (X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()
    
    classes={0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",
             4:"Coat",5: "Sandal",6:"Shirt",7:"Sneaker",8:"Bag",
             9:"Ankle boot"}
    
    num_of_examples=X_train.shape[0]
    l=list(range(num_of_examples))
    r.shuffle(l)
    train_x=np.empty((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    train_y=np.empty(y_train.shape)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

    for i in range(num_of_examples):
        train_x[i]=X_train[l[i]].flatten()
        train_y[i]=y_train[l[i]]
    chosen_image_class=dict()
    j=0
    examples=[]
    while len(chosen_image_class)<10:
        if train_y[j] not in chosen_image_class:
            chosen_image_class[train_y[j]]=None
            img=wandb.Image(np.resize(train_x[j],X_train.shape[1:]),caption=classes[train_y[j]])
            examples.append(img) 
        j+=1
    
    wandb.log({"Examples":examples})
    
    
    if scaling=="standard_normal":
        train_x=(train_x - np.mean(train_x,axis=0))/np.std(train_x,axis=0)
        X_test=(X_test - np.mean(X_test,axis=0))/np.std(X_test,axis=0)
        
    elif scaling=="MinMax":
        train_x=(train_x - np.min(train_x,axis=0))/(np.max(train_x,axis=0)-np.min(train_x,axis=0))
        X_test=(X_test - np.min(X_test,axis=0))/(np.max(X_test,axis=0)-np.min(X_test,axis=0))

    wandb.finish()
    return (train_x,train_y),(X_test,y_test)

data=load_fashion_mnist()

seed_value=0
def hyperparametric_tuning_1():
    global seed_value
    config_defaults={
        "dataset": "fashion_mnist",
        "epochs": 5,
        "batch_size": 32,
        "num_layers": 32,
        "hidden_size": 64,
        "learning_rate": 10**-3,
        "momentum": 0.9,
        "beta": 0.9,
        "beta1":0.99,
        "beta2":0.999,
        "epsilon":10**-8,
        "optimizer": "Adam",
        "weight_decay": 0.005,
        "weight_init": "Xavier",
        "activation": "sigmoid",
        "loss": "cross_entropy",
        "leakyalpha":0.1,
        "val_percent": 0.1,
        "num_of_classes": 10,
        "early_stopping": False,
        "patience": 3,
        "data_augmentation": False,       
    }
    
    run=wandb.init(project="cs6910_assignment_1",config=config_defaults)
    config=wandb.config
    sweep_name="ep_{}_bs_{}_hlnum_{}_hlsize_{}_lr_{}_opt_{}_init_{}_act_{}_loss_{}_l2_{}".format(config.epochs,config.batch_size,
                                                                                                 config.num_layers,
                                                                                                 config.hidden_size,config.learning_rate,
                                                                                                 config.optimizer,config.weight_init,
                                                                                                 config.activation,config.loss,
                                                                                                 config.weight_decay
                                                                                                 )
    
    run.name=sweep_name
    wandb.log({"Random_seed":seed_value})
    (X_train,y_train),(X_test,y_test)=data
    input_dimension = X_train[0].shape[0]
    num_of_classes=config.num_of_classes
    num_of_neurons_list = [config.hidden_size]*(config.num_layers) +[num_of_classes]
    activation_func_list = ([config.activation]*(config.num_layers) + ["softmax"]) if config.activation!="leakyreLU" else ([(config.activation,config.leakyalpha)]*(config.num_layers) + ["softmax"])


    model=nn_model()
    model.compact_build_nn(input_dimension,config.num_layers+1,num_of_neurons_list,activation_func_list,
                       config.weight_init,seed_value)
    
    model.training(X_train,y_train,val_percent=config.val_percent,loss_func= config.loss,optimizer=config.optimizer,
                  max_epoch=config.epochs,batch_size=config.batch_size,learning_rate=config.learning_rate,
                  beta=config.momentum,rmsbeta=config.beta,beta_1=config.beta1,beta_2=config.beta2,epsilon=config.epsilon,L2_decay=config.weight_decay,
                  early_stopping=config.early_stopping,patience=config.patience,data_augmentation=config.data_augmentation,seed=seed_value)
    seed_value+=1

#1st sweep configuration: random search with run cap 300
sweep_configuration_1={
    "project": "cs6910_assignment_1",
    "method" : "random",
    "name" : "hyperparameter_tuning_1",
    "metric": {
        "goal": "maximize",
        "name": "val_accuracy"
    },
    "run_cap":300,
    "parameters":{
        "epochs": {"values": [5,10,15],
                   "probabilities":[0.3,0.5,0.2]},
        "batch_size": {"values": [16,32,64]},
        "num_layers": {"values": [3,4,5],
                       "probabilities":[0.15,0.35,0.5]},
        "hidden_size": {"values": [32,64,128],
                        "probabilities": [0.2,0.3,0.5]},
        "learning_rate": {"values": [10**-3,10**-4]},
        "optimizer": {"values":
                     ["SGD","momentumGD","Nesterov",
                     "RMSprop","Adam","NAdam"],
                      "probabilities":[0.1,0.1,0.15,0.15,0.25,0.25]
                     },
        "weight_decay": {"values": [0,0.5,0.05,0.005]
                    },
        "weight_init": {"values": 
                                  ["random","Xavier"]
                                 },
        "activation": {"values" : 
                                ["sigmoid","tanh", "reLU","leakyreLU"],
                                "probabilities":[0.15,0.15,0.35,0.35]
                               },
        "loss": {"values" : 
                      ["cross_entropy"]
                      }  
    }
}
sweep_id= wandb.sweep(sweep_configuration_1,project="cs6910_assignment_1")
wandb.agent(sweep_id,function=hyperparametric_tuning_1)

#second_configuration: grid sweep of total runs 144
seed_value=400
sweep_configuration_2={
    "project": "cs6910_assignment_1",
    "method" : "grid",
    "name" : "hyperparameter_tuning_2",
    "metric": {
        "goal": "maximize",
        "name": "val_accuracy"
    },
    "parameters":{
        "epochs": {"values": [15,20]},
        "batch_size": {"values": [64]},
        "num_layers": {"values": [5]},
        "hidden_size": {"values": [64,128]},
        "learning_rate": {"values": [0.0025,0.005,0.0075]},
        "optimizer": {"values":
                     ["Adam","NAdam"],
                     },
        "weight_decay": {"values": [0.05,0.005,0.001]
                    },
        "weight_init": {"values": 
                                  ["Xavier"]
                                 },
        "activation": {"values" : 
                                ["reLU","leakyreLU"],
                               },
        "loss": {"values" : 
                      ["cross_entropy"]
                      }  
    }
}
sweep_id= wandb.sweep(sweep_configuration_2,project="cs6910_assignment_1")
wandb.agent(sweep_id,function=hyperparametric_tuning_1)





