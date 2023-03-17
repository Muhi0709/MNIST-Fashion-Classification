from NN_model import nn_model
from data_loader import get_mnist
import wandb

wandb.login()
data=get_mnist()

def train_test_mnist(config_defaults,seed,name):

    wandb.init(project="cs6910_assignment_1",config=config_defaults,name=name)
    config=wandb.config
    (X_train,y_train),(X_test,y_test)=data
    input_dimension = X_train[0].shape[0]
    num_of_classes=config.num_of_classes
    num_of_neurons_list = [config.hidden_size]*(config.num_layers) +[num_of_classes]
    activation_func_list = ([config.activation]*(config.num_layers) + ["softmax"]) if config.activation!="leakyreLU" else ([(config.activation,config.leakyalpha)]*(config.num_layers) + ["softmax"])


    best_model=nn_model()
    best_model.compact_build_nn(input_dimension,config.num_layers+1,num_of_neurons_list,activation_func_list,
                       config.weight_init,seed)
    
    best_model.training(X_train,y_train,val_percent=config.val_percent,loss_func= config.loss,optimizer=config.optimizer,
                  max_epoch=config.epochs,batch_size=config.batch_size,learning_rate=config.learning_rate,
                  beta=config.momentum,rmsbeta=config.beta,beta_1=config.beta1,beta_2=config.beta2,epsilon=config.epsilon,L2_decay=config.weight_decay,
                  early_stopping=config.early_stopping,patience=config.patience,data_augmentation=config.data_augmentation,seed=seed)
    wandb.finish()

    acc,loss=best_model.compute_loss_accuracy(X_test,y_test)
    print("-"*100)
    print("=>Test Accuracy:",acc)
    print("=>Test Loss:",loss)

config_1={}
config_2={}
config_3={}

print("Test-1")
train_test_mnist(config_1,0,"test_1")
print("Test-2")
train_test_mnist(config_2,0,"test_2")
print("Test-3",config_3,0,"test_3")






