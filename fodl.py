from keras.datasets import fashion_mnist
import wandb
import numpy as np

wandb.login()

class nn_model():
    W={0:None}
    b={0:None}
    a={0:None}
    h={0:None}
    grad_W={0:None}
    grad_b={0:None}
    grad_a={0:None}
    grad_h={0:None}
    activations={0:None}
    layer_type={0:None}
    layers_count=0
    last_layer_neuron_count=None
    
    def add_input_layer(self,ndim):
        self.h[0]=np.empty(ndim)
        self.layer_type[0]="input"
        self.last_layer_neuron_count=ndim

    
    def add_layer(self,num_neurons,layer_type="hidden",activation_func="sigmoid",w_initialization="random"):
        layer_idx=self.layers_count+1
        
        if w_initialization=="Xavier":
            n=self.last_layer_neuron_count
            self.W[layer_idx]=np.random.normal(0,1/np.power(n,0.5),(num_neurons,self.last_layer_neuron_count))
        elif w_initialization=="normalized_Xavier":
            n=self.last_layer_neuron_count
            m=num_neurons
            self.W[layer_idx]=np.random.normal(0,np.power(2/(n+m),0.5),(num_neurons,self.last_layer_neuron_count))
            
        elif w_initialization=="He":
            n=self.last_layer_neuron_count
            mu=0
            sd=(2/n)**0.5
            self.W[layer_idx]=np.random.normal(mu,sd,(num_neurons,self.last_layer_neuron_count))
        else:
            min=-0.1
            max=0.1
            self.W[layer_idx]=np.random.uniform(min,max,(num_neurons,self.last_layer_neuron_count))
            
        self.b[layer_idx]=np.zeros(num_neurons)

        self.a[layer_idx]=np.zeros(num_neurons)
        self.h[layer_idx]=np.zeros(num_neurons)
        self.grad_W[layer_idx]=np.zeros((num_neurons,self.last_layer_neuron_count))
        self.grad_b[layer_idx]=np.zeros(num_neurons)
        self.grad_a[layer_idx]=np.zeros(num_neurons)
        self.grad_h[layer_idx]=np.zeros(num_neurons)
        self.activations[layer_idx]=activation_func
        self.layer_type[layer_idx]=layer_type
        self.last_layer_neuron_count=num_neurons
        self.layers_count+=1
    
    def compact_build_nn(self,input_layer_dim,num_of_layers,num_of_neurons_list,activation_func_list,initialization):
        self.add_input_layer(input_layer_dim)
        for i in range(num_of_layers):
            self.add_layer(num_of_neurons_list[i],"hidden" if i<num_of_layers-1 else "output",activation_func_list[i],
                           initialization)
    
    
    
    def forward_propagation(self,input_x,calculate_error=False,input_y=0,loss_func="cross_entropy"):
        layer_count=self.layers_count
        error=None
        correct=0

    
        for i in range(1,layer_count+1):
            self.a[i]= self.W[i]@ (input_x if i==1 else self.h[i-1]) + self.b[i]
            if self.activations[i]=="sigmoid":
                self.h[i]=1/(1+np.exp(-1*self.a[i]))
            elif self.activations[i]=="tanh":
                self.h[i]=np.tanh(self.a[i])
            
            elif self.activations[i]=="reLU":
                self.h[i]=np.maximum(self.a[i],np.zeros(self.a[i].shape))

            elif self.activations[i]=="softmax":
                self.h[i]=np.exp(self.a[i])/np.sum(np.exp(self.a[i]))

        maxi=np.argmax(input_y)
        correct=1 if np.argmax(self.h[layer_count])==maxi else 0
        

        if loss_func=="mse" and calculate_error==True:
            error=(np.linalg.norm(self.h[layer_count]-input_y))**2
        elif loss_func=="cross_entropy" and calculate_error==True:
            error=-1*np.log2(self.h[layer_count][maxi])
            

        return correct,error

    def back_propagation(self,input_y,L2_decay=0,loss_func="cross_entropy",N=1):
        
        L=self.layers_count
        
        if self.activations[L]=="softmax" and loss_func=="cross_entropy":
            self.grad_a[L]= 1/N * (self.h[L]-input_y)
            
        elif self.activations[L]=="softmax" and loss_func=="mse":
            k=self.last_layer_neuron_count
            y_hat=self.h[L]
            y_hat_diag=np.diag(y_hat)
            self.grad_a[L]= (2/N) * (y_hat_diag - np.outer(y_hat,y_hat)) @ (y_hat-input_y)
        
        for i in range(L,0,-1):
            self.grad_h[i-1]= self.W[i].T @ self.grad_a[i]
            self.grad_W[i]+= np.outer(self.grad_a[i],self.grad_h[i-1]) + ((2/N)*L2_decay*self.W[i])
            self.grad_b[i]+= self.grad_a[i]
            
            if i==1:
                continue
                
            if self.activations[i-1]=="sigmoid":
                self.grad_a[i-1]= self.grad_h[i-1] * (self.a[i-1] -self.a[i-1]**2)
                
            elif self.activations[i-1]=="tanh":
                self.grad_a[i-1]=self.grad_h[i-1] * (1- (self.a[i-1])**2)
            
            elif self.activations[i-1]=="reLU":
                self.grad_a[i-1]= self.grad_h[i-1] * (np.where(self.a[i-1]>0,np.ones(self.a[i-1].shape),
                                                      np.zeros(self.a[i-1].shape)))
            
            
            
    def validation(self,val_x,val_y,loss_func):
        val_accuracy=0
        val_loss=0
        num_of_classes = self.last_layer_neuron_count
        L=self.layers_count
        
        for i in range(len(val_x)):
            self.forward_propagation(val_x[i])
            y_pred= self.h[L]
            maxi=np.argmax(val_y[i])
            val_accuracy+=  (1 if np.argmax(y_pred)==maxi else 0)
            if loss_func=="cross_entropy":
                val_loss+= - np.log2(y_pred[maxi])
            else:
                val_loss+= np.linalg.norm(y_pred-val_y)**2
            
        val_accuracy=val_accuracy *100 /len(val_x)
        val_loss=(val_loss/len(val_x))
        
        return val_accuracy,val_loss
    
    
    def gradient_descent(self,batch_size,max_epoch,learning_rate,training_x,training_y,val_x,val_y,l2_decay,loss_func):
        num_of_layers=self.layers_count
        N=len(training_x)
        
        for epoch in range(1,max_epoch+1):
            count=0
            training_loss=0
            training_accuracy=0
            val_loss=0
            val_accuracy=0
            
            for j in range(len(training_x)):
                acc,err=self.forward_propagation(training_x[j],True,training_y[j],loss_func)
                training_accuracy+=acc
                training_loss+=err
                self.back_propagation(training_y[j],l2_decay,loss_func,N)
                count+=1
                if count == batch_size or j==len(training_x):
                    
                    for i in range(1,num_of_layers+1):
                        self.W[i] = self.W[i] - learning_rate * self.grad_W[i]
                        self.b[i] = self.b[i] - learning_rate * self.grad_b[i]
                        
                    count=0
                    for k in range(1,num_of_layers+1):
                        self.grad_W[k] = np.zeros(self.grad_W[k].shape)
                        self.grad_b[k] = np.zeros(self.grad_b[k].shape)
            
            
            training_loss=training_loss/N 
            training_accuracy=(training_accuracy * 100)/N                
            val_accuracy,val_loss=self.validation(val_x,val_y,loss_func)
            
            
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                      "val_accuracy":val_accuracy,
                      "val_loss": val_loss})
            
            
            
                            
    def momentum_gd(self,batch_size,max_epoch,learning_rate,momentum_parameter,training_x,training_y,val_x,val_y,l2_decay,loss_func):
        
        num_of_layers = self.layers_count
        N=len(training_x)
        
        u_weights={i:0 for i in range(1,num_of_layers+1)}
        u_bias={i:0 for i in range(1,num_of_layers+1)}
        
        for epoch in range(1,max_epoch+1):
            count=0
            training_loss=0
            training_accuracy=0
            val_loss=0
            val_accuracy=0
            
            for j in range(len(training_x)):

                acc,err=self.forward_propagation(training_x[j],True,training_y[j],loss_func)
                training_accuracy+=acc
                training_loss+=err
                self.back_propagation(training_y[j],l2_decay,loss_func,N)
                count+=1
                if count == batch_size or j==len(training_x):
                    for i in range(1,num_of_layers+1):
                        u_weights[i]= momentum_parameter*u_weights[i] + learning_rate * self.grad_W[i] 
                        u_bias[i]= momentum_parameter*u_bias[i] + learning_rate * self.grad_b[i]

                        self.W[i] = self.W[i] - u_weights[i]
                        self.b[i] = self.b[i] - u_bias[i]
                        
                    count=0
                    for k in range(1,num_of_layers+1):
                        self.grad_W[k] = np.zeros(self.grad_W[k].shape)
                        self.grad_b[k] = np.zeros(self.grad_b[k].shape)
                            
                            
            
            training_loss=training_loss/N    
            training_accuracy=(training_accuracy * 100)/N                
            val_accuracy,val_loss=self.validation(val_x,val_y,loss_func)
            


            
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                       "val_accuracy":val_accuracy,
                       "val_loss": val_loss})
            
            
          
    
    def NAG(self,batch_size,max_epoch,learning_rate,momentum_parameter,training_x,training_y,val_x,val_y,l2_decay,loss_func):
        
        num_of_layers = self.layers_count
        N=len(training_x)
        
        u_weights={i:0 for i in range(1,num_of_layers+1)}
        u_bias={i:0 for i in range(1,num_of_layers+1)}
        
        for epoch in range(1,max_epoch+1):
            count=0
            training_loss=0
            training_accuracy=0
            val_loss=0
            val_accuracy=0
            
            for j in range(len(training_x)):
            
                acc,err=self.forward_propagation(training_x[j],True,training_y[j],loss_func)
                training_accuracy+=acc
                training_loss+=err
                self.back_propagation(training_y[j],l2_decay,loss_func,N)
                count+=1
                
                if count == batch_size or j==len(training_x):
                    
                    for i in range(1,num_of_layers+1):
                        self.W[i] = self.W[i] + momentum_parameter*u_weights[i]
                        self.b[i] = self.b[i] + momentum_parameter*u_bias[i]
                        
                        u_weights[i]= momentum_parameter*u_weights[i] + learning_rate*self.grad_W[i]
                        u_bias[i]= momentum_parameter*u_bias[i] + learning_rate*self.grad_b[i]
                        
                        self.W[i] = self.W[i]  - u_weights[i]
                        self.b[i] = self.b[i] - u_bias[i]
                        
                    count=0
                    for k in range(1,num_of_layers+1):
                        self.grad_W[k] = np.zeros(self.grad_W[k].shape)
                        self.grad_b[k] = np.zeros(self.grad_b[k].shape)
                        
                        self.W[k]=self.W[k] - momentum_parameter*u_weights[k]
                        self.b[k]=self.b[k] - momentum_parameter*u_bias[k]
                            
                            
            
            training_loss=training_loss/N    
            training_accuracy=(training_accuracy * 100)/N                
            val_accuracy,val_loss=self.validation(val_x,val_y,loss_func)


                    
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                       "val_accuracy":val_accuracy,
                       "val_loss": val_loss})
            
            

            
    def AdaGrad_RMSProp(self,batch_size,max_epoch,learning_rate,beta,eps,training_x,training_y,val_x,val_y,l2_decay,loss_func,option):
        num_of_layers = self.layers_count
        N=len(training_x)
        v_weights={i:0 for i in range(1,num_of_layers+1)}
        v_bias={i:0 for i in range(1,num_of_layers+1)}
        
        for epoch in range(1,max_epoch+1):
            count=0
            training_loss=0
            training_accuracy=0
            val_loss=0
            val_accuracy=0
            for j in range(len(training_x)):
              
                acc,err=self.forward_propagation(training_x[j],True,training_y[j],loss_func)
                training_accuracy+=acc
                training_loss+=err
                self.back_propagation(training_y[j],l2_decay,loss_func,N)
                count+=1
                if count == batch_size or j==len(training_x):
                    for i in range(1,num_of_layers+1):
                        v_weights[i]= (v_weights[i] + np.power(self.grad_W,2)) if option==0 else beta*v_weights[i] + (1-beta)*(np.power(self.grad_W[i],2))
                        v_bias[i]= (v_bias[i] + np.power(self.grad_b,2)) if option==0 else beta*v_bias[i] + (1-beta)*np.power(self.grad_b[i],2)
                        
                        self.W[i] = self.W[i] - (learning_rate/(np.power(v_weights[i]+eps,0.5))) * self.grad_W[i]
                        self.b[i] = self.b[i] - (learning_rate/(np.power(v_bias[i]+eps,0.5))) * self.grad_b[i]
                        
                    count=0
                    for k in range(1,num_of_layers+1):
                        self.grad_W[k] = np.zeros(self.grad_W[k].shape)
                        self.grad_b[k] = np.zeros(self.grad_b[k].shape)
            
            
            training_loss=training_loss/N    
            training_accuracy=(training_accuracy * 100)/N                
            val_accuracy,val_loss=self.validation(val_x,val_y,loss_func)

            
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                       "val_accuracy":val_accuracy,
                       "val_loss": val_loss})
            
        
    def Adam_NAdam(self,batch_size,max_epoch,learning_rate,beta_1,beta_2,eps,
                   training_x,training_y,val_x,val_y,l2_decay,loss_func,option):
        
        num_of_layers=self.layers_count
        N=len(training_x)
        t=1
        v_weights={i:0 for i in range(1,num_of_layers+1)}
        v_bias={i:0 for i in range(1,num_of_layers+1)}
        
        m_weights={i:0 for i in range(1,num_of_layers+1)}
        m_bias={i:0 for i in range(1,num_of_layers+1)}
        
        for epoch in range(1,max_epoch+1):
            count=0
            training_loss=0
            training_accuracy=0
            val_loss=0
            val_accuracy=0
            
            for j in range(len(training_x)):
                acc,err=self.forward_propagation(training_x[j],True,training_y[j],loss_func)
                training_accuracy+=acc
                training_loss+=err
                self.back_propagation(training_y[j],l2_decay,loss_func,N)
                count+=1
                if count == batch_size or j==len(training_x):
                    
                    for i in range(1,num_of_layers+1):
                        m_weights[i] = beta_1* m_weights[i] + (1- beta_1)* self.grad_W[i]
                        m_bias[i] = beta_1 * m_bias[i] + (1- beta_1) * self.grad_b[i]
                        
                        v_weights[i] = beta_2* v_weights[i] + (1- beta_2)* np.power(self.grad_W[i],2)
                        v_bias[i] = beta_2* v_bias[i] + (1- beta_2)* np.power(self.grad_b[i],2)
                        
                        m_weights_hat= m_weights[i]/(1- beta_1**t)
                        m_bias_hat=m_bias[i]/(1- beta_1**t)
                        v_weights_hat = v_weights[i]/(1-beta_2**t)
                        v_bias_hat = v_bias[i]/(1- beta_2 ** t)
                        
                        if option==1:
                            self.W[i] = self.W[i] - ((learning_rate/(np.power(v_weights_hat,0.5)+eps)) \
                            *(beta_1* m_weights_hat + (((1-beta_1)*self.grad_W[i])/(1-beta_1**t))))
                            
                            self.b[i] = self.b[i] - ((learning_rate/(np.power(v_bias_hat,0.5)+eps))\
                            *(beta_1* m_bias_hat + (((1-beta_1)*self.grad_b[i])/(1-beta_1**t))))
                            
                        self.W[i] = self.W[i] - (learning_rate/(np.power(v_weights_hat,0.5)+eps)) * m_weights_hat
                        self.b[i] = self.b[i] - (learning_rate/(np.power(v_bias_hat,0.5)+eps)) * m_bias_hat
                        
                    t+=1    
                    count=0
                    for k in range(1,num_of_layers+1):
                        self.grad_W[k] = np.zeros(self.grad_W[k].shape)
                        self.grad_b[k] = np.zeros(self.grad_b[k].shape)
            
            
            training_loss=training_loss/N    
            training_accuracy=(training_accuracy * 100)/N                
            val_accuracy,val_loss=self.validation(val_x,val_y,loss_func)
            
          
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                       "val_accuracy":val_accuracy,
                       "val_loss": val_loss})
            
            
            

                            
    def training(self,x,y,val_percent=0.1,loss_func="cross_entropy",optimizer="GD",
                 max_epoch=10,batch_size=1,learning_rate=0.02,beta=0.9,beta_1=0.99,beta_2=0.999,epsilon=10**-6,L2_decay=0):
        
        num_of_classes=self.last_layer_neuron_count
        num_of_training_examples= int(len(x)*(1-val_percent))
        data=np.vstack((x.T,np.array([y]))).T
        np.random.shuffle(data)
        
        training_x=data[:num_of_training_examples,:-1]
        val_x=data[num_of_training_examples:,:-1]
        
        one_hot_encoded_labels=np.zeros((len(data[:,-1]),num_of_classes))
        for i in range(len(y)):
            one_hot_encoded_labels[i][int(data[:,-1][i])]=1
        
        encoded_training_y=one_hot_encoded_labels[:num_of_training_examples]
        encoded_val_y= one_hot_encoded_labels[num_of_training_examples:]

 
        if optimizer=="GD":
            self.gradient_descent(len(training_x),max_epoch,learning_rate,training_x,encoded_training_y,val_x,
                                  encoded_val_y,L2_decay,loss_func)
        
        elif optimizer=="SGD":
            self.gradient_descent(1,max_epoch,learning_rate,training_x,encoded_training_y,val_x,encoded_val_y,L2_decay,loss_func)
            
        elif optimizer=="Mini_batch":
            self.gradient_descent(batch_size,max_epoch,learning_rate,training_x,encoded_training_y,val_x,
                                  encoded_val_y,L2_decay,loss_func)
            
        elif optimizer=="momentumGD":
            self.momentum_gd(batch_size,max_epoch,learning_rate,beta,training_x,
                             encoded_training_y,val_x,encoded_val_y,L2_decay,loss_func)
            
        elif optimizer=="Nesterov":
            self.NAG(batch_size,max_epoch,learning_rate,beta,training_x,encoded_training_y,val_x,
                     encoded_val_y,L2_decay,loss_func)
            
        elif optimizer=="Adagrad":
            self.AdaGrad_RMSprop(batch_size,max_epoch,learning_rate,beta,epsilon,training_x,encoded_training_y,val_x,encoded_val_y,
                                 L2_decay,loss_func,0)
        
        elif optimizer=="RMSprop":
            self.AdaGrad_RMSProp(batch_size,max_epoch,learning_rate,beta,epsilon,training_x,encoded_training_y,
                                 val_x,encoded_val_y,L2_decay,loss_func,1)
        elif optimizer=="Adam":
            self.Adam_NAdam(batch_size,max_epoch,learning_rate,beta_1,beta_2,epsilon,
                   training_x,encoded_training_y,val_x,encoded_val_y,L2_decay,loss_func,0)
        elif optimizer=="NAdam":
            self.Adam_NAdam(batch_size,max_epoch,learning_rate,beta_1,beta_2,epsilon,
                   training_x,encoded_training_y,val_x,encoded_val_y,L2_decay,loss_func,1)
            
    
   
        
    def predict(self,data_x,encoded=False):
        num_of_classes = self.last_layer_neuron_count
        L=self.layers_count
        y_class_predicted=[]
        for i in len(data_x):
            self.forward_propagation(data_x)
            if encoded:
                encoded_y=np.zeros(num_of_classes)
                encoded_y[np.argmax(self.h[L])]=1
                y_class_predicted.append(encoded_y)
            y_class_predicted.append(np.argmax(self.h[L]))
            
        return np.array(y_class_predicted)

def load_fashion_mnist(scaling="standard"):
    wandb.init("project-1")
    (X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()
    classes={0:"T-shirt/top",1:"Trouser",2:"Pullover",3:"Dress",
             4:"Coat",5: "Sandal",6:"Shirt",7:"Sneaker",8:"Bag",
             9:"Ankle boot"}
    
    num_of_examples=X_train.shape[0]
    l=list(range(num_of_examples))
    np.random.shuffle(l)
    train_x=np.empty((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    train_y=np.empty(y_train.shape)
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
    
    if scaling=="standard":
      X_train=(train_x - np.mean(train_x,axis=0))/np.std(train_x,axis=0)
      X_test (X_test - np.mean(X_test,axis=0))/np.std(X_test,axis=0)
      
    elif scaling=="MinMax":
      X_train=(train_x - np.min(train_x,axis=0))/(np.max(train_x,axis=0)-np.min(train_x,axis=0))
      X_test=(X_test - np.min(X_test,axis=0))/(np.max(X_test,axis=0)-np.min(X_test,axis=0))
    
    return (X_train,train_y),(X_test,y_test)


data=load_fashion_mnist()
sweep_configuration={
    "project": "project-1",
    "method" : "random",
    "name" : "hyperparameter_tuning1",
    "metric": {
        "goal": "maximize",
        "name": "val_accuracy"
    },
    "parameters":{
        "max_epoch": {"values": [5,10]},
        "batch_size": {"values": [16,32,64]},
        "num_of_hidden_layers": {"values": [3,4,5]},
        "hidden_layer_size": {"values": [32,64,128]},
        "learning_rate": {"values": [10**-3,10**-4]},
        "optimizer": {"values":
                     ["SGD","momentumGD","Nesterov",
                     "RMSprop","Adam","NAdam"]
                     },
        "L2_decay": {"values": [0,0.5,0.0005]
                    },
        "weight_initialization": {"values": 
                                  ["random","normalized_Xavier"]
                                 },
        "activation_function": {"values" : 
                                ["sigmoid","tanh", "reLU"]
                               },
        "loss_func": {"values" : 
                      ["cross_entropy"]
                      }  
    }
}

def hyperparametric_tuning1():

    config_defaults={
        "max_epoch": 5,
        "batch_size": 32,
        "num_of_hidden_layers": 32,
        "hidden_layer_size": 64,
        "learning_rate": 10**-3,
        "beta": 0.9,
        "beta_1":0.99,
        "beta_2":0.999,
        "eps":10**-8,
        "optimizer": "Adam",
        "L2_decay": 0.005,
        "weight_initialization": "normalized_Xavier",
        "activation_function": "sigmoid",
        "loss_func": "cross_entropy",
        "val_percent": 0.1,
        "num_of_classes": 10
    }
    

    run=wandb.init(project="project-1",config=config_defaults)
    config=wandb.config

    sweep_name="ep_{}_bs_{}_hlnum_{}_hlsize_{}_lr_{}_opt_{}_init_{}_act_{}_loss_{}_l2_{}".format(config.max_epoch,config.batch_size,
    config.num_of_hidden_layers,config.hidden_layer_size,config.learning_rate,config.optimizer,config.weight_initialization,
    config.activation_function,config.loss_func,config.L2_decay)
    
    run.name=sweep_name
    (X_train,y_train),(X_test,y_test)=data
    input_dimension = X_train[0].shape[0]
    num_of_classes=config.num_of_classes
    num_of_neurons_list = [config.hidden_layer_size]*(config.num_of_hidden_layers) +[num_of_classes]
    activation_func_list = [config.activation_function]*(config.num_of_hidden_layers) + ["softmax"]
    
    model=nn_model()
    model.compact_build_nn(input_dimension,config.num_of_hidden_layers+1,num_of_neurons_list,activation_func_list,
                       config.weight_initialization)
    
    model.training(X_train,y_train,val_percent=0.1,loss_func= config.loss_func,optimizer=config.optimizer,
                  max_epoch=config.max_epoch,batch_size=config.batch_size,learning_rate=config.learning_rate,
                  beta=config.beta,beta_1=config.beta_1,beta_2=config.beta_2,epsilon=config.eps,L2_decay=config.L2_decay)
    

sweep_id= wandb.sweep(sweep_configuration,project="project-1")
wandb.agent(sweep_id,function=hyperparametric_tuning1,count=100)