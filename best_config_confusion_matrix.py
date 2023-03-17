from NN_model import nn_model
import numpy as np
from data_loader import get_fashion_mnist
import plotly.express as px
import wandb
wandb.login()
data=get_fashion_mnist()

seed_value=0
def train_test_best_configuration():
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
    sweep_name="best_ep_{}_bs_{}_hlnum_{}_hlsize_{}_lr_{}_opt_{}_init_{}_act_{}_loss_{}_l2_{}_early_{}_patie_{}_augment_{}".format(config.epochs,config.batch_size,
                                                                                                 config.num_layers,
                                                                                                 config.hidden_size,config.learning_rate,
                                                                                                 config.optimizer,config.weight_init,
                                                                                                 config.activation,config.loss,
                                                                                                 config.weight_decay,config.early_stopping,
                                                                                                 config.patience,
                                                                                                 config.data_augmentation
                                                                                                 )
    
    run.name=sweep_name
    (X_train,y_train),(X_test,y_test)=data
    input_dimension = X_train[0].shape[0]
    num_of_classes=config.num_of_classes
    num_of_neurons_list = [config.hidden_size]*(config.num_layers) +[num_of_classes]
    activation_func_list = ([config.activation]*(config.num_layers) + ["softmax"]) if config.activation!="leakyreLU" else ([(config.activation,config.leakyalpha)]*(config.num_layers) + ["softmax"])


    best_model=nn_model()
    best_model.compact_build_nn(input_dimension,config.num_layers+1,num_of_neurons_list,activation_func_list,
                       config.weight_init,seed_value)
    
    best_model.training(X_train,y_train,val_percent=config.val_percent,loss_func= config.loss,optimizer=config.optimizer,
                  max_epoch=config.epochs,batch_size=config.batch_size,learning_rate=config.learning_rate,
                  beta=config.momentum,rmsbeta=config.beta,beta_1=config.beta1,beta_2=config.beta2,epsilon=config.epsilon,L2_decay=config.weight_decay,
                  early_stopping=config.early_stopping,patience=config.patience,data_augmentation=config.data_augmentation,seed=seed_value)
    wandb.finish()

    y_predicted=best_model.predict(X_test)
    acc,loss=best_model.compute_loss_accuracy(X_test,y_test)
    print("-"*100)
    print("=>Test Accuracy:",acc)
    print("=>Test Loss:",loss)
    return best_model,y_predicted

def plot_confusion_matrix(y_predicted,y_actual,num_of_classes=10,title="Confusion Matrix : Fashion MNist Test Data",
                          labels= ["Top","Trouser","Pullover","Dress","Coat",
                                   "Sandal","Shirt","Sneaker","Bag","Ankle boot"]):
    
    run=wandb.init(project="cs6910_assignment_1")
    run.name="Con_mat_{}".format(title)
    confusion_matrix=np.zeros((num_of_classes,num_of_classes))
    for i in range(len(y_predicted)):
        confusion_matrix[y_predicted][y_actual]+=1
    
    fig = px.imshow(confusion_matrix,
                labels=dict(x="Actual Labels", y="True Labels", color="Count"),
                x=labels,
                y=labels,color_continuous_scale="viridis",text_auto=True
               )
    fig.update_xaxes(side="top")
    wandb.log({title: fig})
    wandb.log({title: wandb.plot.confusion_matrix(probs=None,
                        y_true=y_actual, preds=y_predicted,
                        class_names=labels)})
    wandb.finish()

best_config,y_predicted=train_test_best_configuration()
y_actual=data[1][1]
plot_confusion_matrix(y_predicted,y_actual)
