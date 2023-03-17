from NN_model import nn_model
from data_loader import get_fashion_mnist
import wandb
wandb.login()
data=get_fashion_mnist

def loss_func_tuning():
    seed_value=0
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

    (X_train,y_train),(_,_)=data
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


sweep_configuration={
    "project": "cs6910_assignment_1",
    "method" : "grid",
    "name" : "hyperparameter_tuning_2",
    "metric": {
        "goal": "maximize",
        "name": "val_accuracy"
    },
    "parameters":{
        "loss": {"values" : 
                      ["cross_entropy","mse"]
                      }  
    }
}
sweep_id= wandb.sweep(sweep_configuration,project="cs6910_assignment_1")
wandb.agent(sweep_id,function=loss_func_tuning)

    