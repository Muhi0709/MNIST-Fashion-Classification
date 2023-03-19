from NN_model import nn_model
from data_loader import get_mnist

print("loading dataset")
data=get_mnist()
print("dataset loaded successfully")

def train_test_mnist(config,seed):
    (X_train,y_train),(X_test,y_test)=data
    input_dimension = X_train[0].shape[0]
    num_of_classes=config["num_of_classes"]
    num_of_neurons_list = [config["hidden_size"]]*(config["num_layers"]) +[num_of_classes]
    activation_func_list = ([config["activation"]]*(config["num_layers"]) + ["softmax"]) if config["activation"]!="leakyreLU" else ([(config["activation"],config["leakyalpha"])]*(config["num_layers"]) + ["softmax"])


    model=nn_model()
    model.compact_build_nn(input_dimension,config["num_layers"]+1,num_of_neurons_list,activation_func_list,
                       config["weight_init"],seed)
    print("Test-{} Configuration_used:".format(seed+1))
    for key,value in config.items():
        print(key,":",value)
    model.training(X_train,y_train,val_percent=config["val_percent"],loss_func= config["loss"],optimizer=config["optimizer"],
                  max_epoch=config["epochs"],batch_size=config["batch_size"],learning_rate=config["learning_rate"],
                  beta=config["momentum"],rmsbeta=config["beta"],beta_1=config["beta1"],beta_2=config["beta2"],epsilon=config["epsilon"],L2_decay=config["weight_decay"],
                  early_stopping=config["early_stopping"],patience=config["patience"],data_augmentation=config["data_augmentation"],seed=seed,wandb_log=False)

    y_test_encoded=model.one_hot_encoding(y_test)
    acc,loss=model.compute_loss_accuracy(X_test,y_test_encoded,config["loss"])
    print("=>{} Test Accuracy:".format(config["dataset"]),acc)
    print("=>{} Test Loss:".format(config["dataset"]),loss)
    print("-"*200)


config_1={
    "dataset": "mnist",
    "epochs": 15,
    "batch_size": 64,
    "num_layers": 5,
    "hidden_size": 64,
    "learning_rate": 2.5 * (10**-4),
    "momentum": 0.9,
    "beta": 0.9,
    "beta1":0.99,
    "beta2":0.999,
    "epsilon":10**-8,
    "optimizer": "nadam",
    "weight_decay": 0.001,
    "weight_init": "Xavier",
    "activation": "leakyreLU",
    "loss": "cross_entropy",
    "leakyalpha":0.1,
    "val_percent": 0.1,
    "num_of_classes": 10,
    "early_stopping": False,
    "patience": 2,
    "data_augmentation": False,       
    }

config_2={
    "dataset": "mnist",
    "epochs": 20,
    "batch_size": 64,
    "num_layers": 5,
    "hidden_size": 64,
    "learning_rate": 2.5 * (10**-4),
    "momentum": 0.9,
    "beta": 0.9,
    "beta1":0.99,
    "beta2":0.999,
    "epsilon":10**-8,
    "optimizer": "nadam",
    "weight_decay": 0.001,
    "weight_init": "Xavier",
    "activation": "leakyreLU",
    "loss": "cross_entropy",
    "leakyalpha":0.1,
    "val_percent": 0.1,
    "num_of_classes": 10,
    "early_stopping": True,
    "patience": 3,
    "data_augmentation": True,       
    }
config_3={
    "dataset": "mnist",
    "epochs": 20,
    "batch_size": 64,
    "num_layers": 5,
    "hidden_size": 128,
    "learning_rate": 7.5 * (10**-4),
    "momentum": 0.9,
    "beta": 0.9,
    "beta1":0.99,
    "beta2":0.999,
    "epsilon":10**-8,
    "optimizer": "adam",
    "weight_decay": 0.001,
    "weight_init": "Xavier",
    "activation": "leakyreLU",
    "loss": "cross_entropy",
    "leakyalpha":0.1,
    "val_percent": 0.1,
    "num_of_classes": 10,
    "early_stopping": True,
    "patience": 3,
    "data_augmentation": True,       
    }

seed1,seed2,seed3=0,1,2
train_test_mnist(config_1,seed1)
train_test_mnist(config_2,seed2)
train_test_mnist(config_3,seed3)








