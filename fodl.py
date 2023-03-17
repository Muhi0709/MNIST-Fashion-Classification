import wandb
import numpy as np
from queue import Queue

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

    def add_layer(self,num_neurons,layer_type="hidden",activation_func="sigmoid",w_initialization="random",seed=0):
        layer_idx=self.layers_count+1
        r=np.random.default_rng(seed=seed)
        if w_initialization=="Xavier":
            n=self.last_layer_neuron_count
            m= num_neurons
            low=-np.sqrt(6/(n+m))
            high=np.sqrt(6/(n+m))
            self.W[layer_idx]=r.uniform(low,high,(num_neurons,self.last_layer_neuron_count))
            self.b[layer_idx]=r.uniform(low,high,num_neurons)
            
        elif w_initialization=="He":
            n=self.last_layer_neuron_count
            mu=0
            sd=(2/n)**0.5
            self.W[layer_idx]=r.normal(mu,sd,(num_neurons,self.last_layer_neuron_count))
            self.b[layer_idx]=r.normal(mu,sd,num_neurons)
            
        else:
            mini=-0.01
            maxi=0.01
            self.W[layer_idx]=r.uniform(mini,maxi,(num_neurons,self.last_layer_neuron_count))
            self.b[layer_idx]=r.uniform(mini,maxi,num_neurons)
                
        self.a[layer_idx]=np.empty(num_neurons)
        self.h[layer_idx]=np.empty(num_neurons)
        self.grad_W[layer_idx]=np.empty((num_neurons,self.last_layer_neuron_count))
        self.grad_b[layer_idx]=np.empty(num_neurons)
        self.grad_a[layer_idx]=np.empty(num_neurons)
        self.grad_h[layer_idx]=np.empty(num_neurons)
        self.activations[layer_idx]=activation_func
        self.layer_type[layer_idx]=layer_type
        self.last_layer_neuron_count=num_neurons
        self.layers_count+=1
    
    def compact_build_nn(self,input_layer_dim,num_of_layers,num_of_neurons_list,activation_func_list,initialization,seed):
        self.add_input_layer(input_layer_dim)
        for i in range(num_of_layers):
            self.add_layer(num_of_neurons_list[i],"hidden" if i<num_of_layers-1 else "output",activation_func_list[i],
                           initialization,seed)
    
    def shuffle_dataset(self,x,y,seed=0):
        rng=np.random.default_rng(seed=seed)
        data = np.vstack((x.T,np.array([y]))).T
        rng.shuffle(data)
        return data[:,:-1],data[:,-1]
    
    def shuffle_batches(self,batch_x,batch_y,seed=0):
        rng=np.random.default_rng(seed=seed)
        l=list(range(len(batch_x)))
        rng.shuffle(l)
        x,y=[],[]
        for i in range(len(batch_x)):
            x.append(batch_x[l[i]])
            y.append(batch_y[l[i]])
        return x,y

    def train_val_split(self,x,y,split=0.1):
        num_of_training_examples= int((1-split)* len(x))
        train_x = x[:num_of_training_examples]
        val_x  = x[num_of_training_examples:]
        train_y = y[:num_of_training_examples]
        val_y  = y[num_of_training_examples:]
        return (train_x,train_y),(val_x,val_y)

    def one_hot_encoding(self,y):
        num_of_classes=self.last_layer_neuron_count
        encoded_y=np.zeros((len(y),num_of_classes))
        for i in range(len(y)):
            encoded_y[i][int(y[i])]=1
        return encoded_y
    
    def segregate_batches(self,x,y,batch_size):
        N=len(x)
        num_of_batches= int(N/batch_size) + (1 if N%batch_size!=0 else 0)
        batched_x=[x[int(i*batch_size):int(batch_size*(i+1))] for i in range(num_of_batches)]
        batched_y=[y[int(i*batch_size):int(batch_size*(i+1))] for i in range(num_of_batches)]
        return batched_x,batched_y
        
    def forward_propagation(self,batch_x):
        layer_count=self.layers_count
        self.h[0]=batch_x
    
        for i in range(1,layer_count+1):
            self.a[i]= (self.W[i]@ (self.h[i-1].T)).T + self.b[i]
            
            if len(self.activations[i])==2 and self.activations[i][0]=="leakyreLU":
                leaky_alpha=self.activations[i][1]
                self.h[i]=np.maximum(self.a[i],leaky_alpha*self.a[i])
                
            elif self.activations[i]=="sigmoid":
                self.h[i]=1/(1+np.exp(-1*self.a[i]))
                
            elif self.activations[i]=="tanh":
                self.h[i]=np.tanh(self.a[i])
            
            elif self.activations[i]=="reLU":
                self.h[i]=np.maximum(self.a[i],np.zeros(self.a[i].shape))
            
            elif self.activations[i]=="softmax":
                self.h[i]=(np.exp(self.a[i]).T/np.sum(np.exp(self.a[i]),axis=1)).T

            else:
                print("unsupported activation function")
                quit()


    def back_propagation(self,batch_y,L2_decay=0,loss_func="cross_entropy",N="10000"):
        L=self.layers_count
        batch_size=len(batch_y)
        
        if self.activations[L]=="softmax" and loss_func=="cross_entropy":
            self.grad_a[L]=  self.h[L]-batch_y
            
        elif self.activations[L]=="softmax" and loss_func=="mse":
            self.grad_a[L]= (self.h[L] - batch_y) * self.h[L] * (1 - self.h[L])

        else:
            print("unsupported loss function")
            quit()
        
        for i in range(L,0,-1):
            self.grad_W[i]= np.einsum("ki,kj->ij",self.grad_a[i],self.h[i-1]) + 2 * L2_decay*self.W[i]
            self.grad_b[i]= np.sum(self.grad_a[i] ,axis=0) +  2* L2_decay * self.b[i]
            if i==1:
                continue
            self.grad_h[i-1]= ((self.W[i].T) @ (self.grad_a[i].T)).T
                                               
            
            if len(self.activations[i-1])==2 and self.activations[i-1][0]=="leakyreLU":
                leaky_alpha=self.activations[i-1][1]
                self.grad_a[i-1]= self.grad_h[i-1] * (np.where(self.a[i-1]>0,np.ones(self.a[i-1].shape),
                                                              leaky_alpha*np.ones(self.a[i-1].shape)))
                
            elif self.activations[i-1]=="sigmoid":
                self.grad_a[i-1]= self.grad_h[i-1] * (self.a[i-1] -self.a[i-1]**2)
                
            elif self.activations[i-1]=="tanh":
                self.grad_a[i-1]=self.grad_h[i-1] * (1- (self.a[i-1])**2)
            
            elif self.activations[i-1]=="reLU":
                self.grad_a[i-1]= self.grad_h[i-1] * (np.where(self.a[i-1]>0,np.ones(self.a[i-1].shape),
                                                      np.zeros(self.a[i-1].shape)))
            else:
                print("unsupported activation function")
                quit()
                
    def compute_loss_accuracy(self,x,y,loss_func):
        accuracy=0
        loss=0
        num_of_classes = self.last_layer_neuron_count
        L=self.layers_count
        
        self.forward_propagation(x)
        y_pred= np.argmax(self.h[L],axis=1)
        maxi=np.argmax(y,axis=1)
        parity=abs(y_pred-maxi)
        accuracy= np.sum(np.where(parity==0,1,0))
        if loss_func=="cross_entropy":
            loss = np.sum(-1*np.log2(self.h[L]) * y)
            
        elif loss_func=="mse":
            loss = np.sum(np.linalg.norm(self.h[L]-y,axis=1)**2)
        
        else:
            print("unsupported loss function")
            quit()

        accuracy=(accuracy *100)/len(x)
        loss=loss/len(x)
        return accuracy,loss
    
    def update_wei_bias_history(self,patience,w_history,b_history):
        if w_history.qsize()<patience+1:
            w_history.put(self.W)
            b_history.put(self.b)
        else:
            w_history.get()
            w_history.put(self.W)
            b_history.get()
            b_history.put(self.b)
                

    def do_early_stopping(self,patience,prev_loss,curr_loss,inc_val_count,w_history,b_history):
        if curr_loss>prev_loss:
            inc_val_count[0]+=1
        else:
            inc_val_count[0]=0
            
        if inc_val_count[0] == patience:
            self.W= w_history.get()
            self.b= b_history.get()
            return True
        return False
    
    def do_data_augmentation(self,X_train,y_train):         
        N=len(X_train)
        dim=len(X_train[0])
        r=np.random.default_rng(seed=44)
        noise=r.normal(0,0.01,(N,dim))
        noised_signal=X_train+noise
        X_train=np.vstack((X_train,noised_signal))
        y_train=np.hstack((y_train,y_train))
        return X_train,y_train
    

    def gradient_descent(self,max_epoch,learning_rate,training_x,training_y,batched_training_x,batched_training_y,val_x,val_y,
                         l2_decay,loss_func,early_stopping,patience,seed):
        
        num_of_layers=self.layers_count
        N=len(training_x)

        weights_history= None
        bias_history= None
        incr_val_loss_count= None
        val_acc_hist= []

        if early_stopping:
            weights_history=Queue()
            weights_history.put(self.W)
            bias_history=Queue()
            bias_history.put(self.h)
            incr_val_loss_count=[0]

        training_accuracy,training_loss,val_accuracy,val_loss,prev_val_loss=0,0,0,float("inf"),float("inf")
            
        for epoch in range(1,max_epoch+1):
            print("epoch-{}".format(epoch))
            print("-"*200)
            for j in range(len(batched_training_x)):
                self.forward_propagation(batched_training_x[j])
                self.back_propagation(batched_training_y[j],l2_decay,loss_func,N)
                for i in range(1,num_of_layers+1):
                    self.W[i] = self.W[i] - learning_rate * self.grad_W[i]
                    self.b[i] = self.b[i] - learning_rate * self.grad_b[i]
                    
            training_accuracy,training_loss= self.compute_loss_accuracy(training_x,training_y,loss_func)            
            val_accuracy,val_loss=self.compute_loss_accuracy(val_x,val_y,loss_func)
            print("=>train_acc={},train_loss={},val_acc={},val_loss={}".format(training_accuracy,training_loss,
                                                                                val_accuracy,val_loss))
            print("-"*200)

            val_acc_hist.append(val_accuracy)
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                      "val_accuracy":val_accuracy,
                      "val_loss": val_loss})
            
            if early_stopping:
                self.update_wei_bias_history(patience,weights_history,bias_history)
                if self.do_early_stopping(patience,prev_val_loss,val_loss,incr_val_loss_count,weights_history,bias_history):
                    print("Early stopping event has occured!!!")
                    print("val_accuracy (before the event): ", val_acc_hist[int(-1*(patience+1))])
                    break
                prev_val_loss=val_loss
            
            batched_training_x,batched_training_y=self.shuffle_batches(batched_training_x,batched_training_y,seed)

               
    def momentum_gd(self,max_epoch,learning_rate,momentum_parameter,training_x,training_y,batched_training_x,batched_training_y,val_x,
                    val_y,l2_decay,loss_func,early_stopping,patience,seed):
        
        num_of_layers = self.layers_count
        N=len(training_x)
                
        weights_history= None
        bias_history= None
        incr_val_loss_count= None
        val_acc_hist= []

        if early_stopping:
            weights_history=Queue()
            weights_history.put(self.W)
            bias_history=Queue()
            bias_history.put(self.h)
            incr_val_loss_count=[0]
                
        training_accuracy,training_loss,val_accuracy,val_loss,prev_val_loss=0,0,0,float("inf"),float("inf")

        u_weights={i:0 for i in range(1,num_of_layers+1)}
        u_bias={i:0 for i in range(1,num_of_layers+1)}
        
        for epoch in range(1,max_epoch+1):
            print("epoch-{}".format(epoch))
            print("-"*200)
            for j in range(len(batched_training_x)):
                self.forward_propagation(batched_training_x[j])
                self.back_propagation(batched_training_y[j],l2_decay,loss_func,N)
                for i in range(1,num_of_layers+1):
                    u_weights[i]= momentum_parameter*u_weights[i] + learning_rate * self.grad_W[i] 
                    u_bias[i]= momentum_parameter*u_bias[i] + learning_rate * self.grad_b[i]

                    self.W[i] = self.W[i] - u_weights[i]
                    self.b[i] = self.b[i] - u_bias[i]
            training_accuracy,training_loss= self.compute_loss_accuracy(training_x,training_y,loss_func)            
            val_accuracy,val_loss=self.compute_loss_accuracy(val_x,val_y,loss_func)
            print("=>train_acc={},train_loss={},val_acc={},val_loss={}".format(training_accuracy,training_loss,
                                                                                val_accuracy,val_loss))
            print("-"*200)
            val_acc_hist.append(val_accuracy)
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                      "val_accuracy":val_accuracy,
                      "val_loss": val_loss})
            
            if early_stopping:
                self.update_wei_bias_history(patience,weights_history,bias_history)
                if self.do_early_stopping(patience,prev_val_loss,val_loss,incr_val_loss_count,weights_history,bias_history):
                    print("Early stopping event has occured!!!")
                    print("val_accuracy (before the event): ", val_acc_hist[int(-1*(patience+1))])
                    break
                prev_val_loss=val_loss
            
            batched_training_x,batched_training_y=self.shuffle_batches(batched_training_x,batched_training_y,seed)
            
            
    def NAG(self,max_epoch,learning_rate,momentum_parameter,training_x,
            training_y,batched_training_x,batched_training_y,val_x,val_y,l2_decay,loss_func,early_stopping,patience,seed):
        
        num_of_layers = self.layers_count
        N=len(training_x)
        
        weights_history= None
        bias_history= None
        incr_val_loss_count= None
        val_acc_hist= []

        if early_stopping:
            weights_history=Queue()
            weights_history.put(self.W)
            bias_history=Queue()
            bias_history.put(self.h)
            incr_val_loss_count=[0]

        training_accuracy,training_loss,val_accuracy,val_loss,prev_val_loss=0,0,0,float("inf"),float("inf")

        u_weights={i:0 for i in range(1,num_of_layers+1)}
        u_bias={i:0 for i in range(1,num_of_layers+1)}
        for epoch in range(1,max_epoch+1):
            print("epoch-{}".format(epoch))
            print("-"*200)
            for j in range(len(batched_training_x)):
                self.forward_propagation(batched_training_x[j])
                self.back_propagation(batched_training_y[j],l2_decay,loss_func,N)
                for i in range(1,num_of_layers+1):
                    self.W[i] = self.W[i] + momentum_parameter*u_weights[i]
                    self.b[i] = self.b[i] + momentum_parameter*u_bias[i]
                        
                    u_weights[i]= momentum_parameter*u_weights[i] + learning_rate*self.grad_W[i]
                    u_bias[i]= momentum_parameter*u_bias[i] + learning_rate*self.grad_b[i]
                        
                    self.W[i] = self.W[i]  - u_weights[i]
                    self.b[i] = self.b[i] - u_bias[i]
                        
                for k in range(1,num_of_layers+1):
                    self.W[k]=self.W[k] - momentum_parameter*u_weights[k]
                    self.b[k]=self.b[k] - momentum_parameter*u_bias[k]
                               
            training_accuracy,training_loss= self.compute_loss_accuracy(training_x,training_y,loss_func)            
            val_accuracy,val_loss=self.compute_loss_accuracy(val_x,val_y,loss_func)
            print("=>train_acc={},train_loss={},val_acc={},val_loss={}".format(training_accuracy,training_loss,
                                                                                val_accuracy,val_loss))
            print("-"*200)
            val_acc_hist.append(val_accuracy)
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                      "val_accuracy":val_accuracy,
                      "val_loss": val_loss})
            
            if early_stopping:
                self.update_wei_bias_history(patience,weights_history,bias_history)
                if self.do_early_stopping(patience,prev_val_loss,val_loss,incr_val_loss_count,weights_history,bias_history):
                    print("Early stopping event has occured!!!")
                    print("val_accuracy (before the event): ", val_acc_hist[int(-1*(patience+1))])
                    break
                prev_val_loss=val_loss
            
            batched_training_x,batched_training_y=self.shuffle_batches(batched_training_x,batched_training_y,seed)

    def AdaGrad_RMSProp(self,max_epoch,learning_rate,rmsbeta,eps,training_x,training_y,batched_training_x,batched_training_y,
                        val_x,val_y,l2_decay,loss_func,option,early_stopping,patience,seed):
        
        num_of_layers = self.layers_count
        N=len(training_x)
        
        weights_history= None
        bias_history= None
        incr_val_loss_count= None
        val_acc_hist= []

        if early_stopping:
            weights_history=Queue()
            weights_history.put(self.W)
            bias_history=Queue()
            bias_history.put(self.h)
            incr_val_loss_count=[0]

        training_accuracy,training_loss,val_accuracy,val_loss,prev_val_loss=0,0,0,float("inf"),float("inf")

        v_weights={i:0 for i in range(1,num_of_layers+1)}
        v_bias={i:0 for i in range(1,num_of_layers+1)}
        for epoch in range(1,max_epoch+1):
            print("epoch-{}".format(epoch))
            print("-"*200)            
            for j in range(len(batched_training_x)):
                self.forward_propagation(batched_training_x[j])
                self.back_propagation(batched_training_y[j],l2_decay,loss_func,N)
 
                for i in range(1,num_of_layers+1):
                    v_weights[i]= (v_weights[i] + np.power(self.grad_W,2))\
                     if option==0 else rmsbeta*v_weights[i] + (1-rmsbeta)*(np.power(self.grad_W[i],2))
                    v_bias[i]= (v_bias[i] + np.power(self.grad_b,2)) \
                     if option==0 else rmsbeta*v_bias[i] + (1-rmsbeta)*np.power(self.grad_b[i],2)
                        
                    self.W[i] = self.W[i] - (learning_rate/(np.power(v_weights[i]+eps,0.5))) * self.grad_W[i]
                    self.b[i] = self.b[i] - (learning_rate/(np.power(v_bias[i]+eps,0.5))) * self.grad_b[i]
                        
            training_accuracy,training_loss= self.compute_loss_accuracy(training_x,training_y,loss_func)            
            val_accuracy,val_loss=self.compute_loss_accuracy(val_x,val_y,loss_func)
            print("=>train_acc={},train_loss={},val_acc={},val_loss={}".format(training_accuracy,training_loss,
                                                                                val_accuracy,val_loss))
            print("-"*200)
            val_acc_hist.append(val_accuracy)
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                      "val_accuracy":val_accuracy,
                      "val_loss": val_loss})
            
            if early_stopping:
                self.update_wei_bias_history(patience,weights_history,bias_history)
                if self.do_early_stopping(patience,prev_val_loss,val_loss,incr_val_loss_count,weights_history,bias_history):
                    print("Early stopping event has occured!!!")
                    print("val_accuracy (before the event): ", val_acc_hist[int(-1*(patience+1))])
                    break
                prev_val_loss=val_loss
                
            batched_training_x,batched_training_y=self.shuffle_batches(batched_training_x,batched_training_y,seed)

    def Adam_NAdam(self,max_epoch,learning_rate,beta_1,beta_2,eps,training_x,training_y,
                   batched_training_x,batched_training_y,val_x,val_y,l2_decay,loss_func,option,early_stopping,patience,seed):
        
        num_of_layers=self.layers_count
        N=len(training_x)
                
        weights_history= None
        bias_history= None
        incr_val_loss_count= None
        val_acc_hist= []

        if early_stopping:
            weights_history=Queue()
            weights_history.put(self.W)
            bias_history=Queue()
            bias_history.put(self.h)
            incr_val_loss_count=[0]

        training_accuracy,training_loss,val_accuracy,val_loss,prev_val_loss=0,0,0,float("inf"),float("inf")
        
        t=1
        v_weights={i:0 for i in range(1,num_of_layers+1)}
        v_bias={i:0 for i in range(1,num_of_layers+1)}
        m_weights={i:0 for i in range(1,num_of_layers+1)}
        m_bias={i:0 for i in range(1,num_of_layers+1)}

        for epoch in range(1,max_epoch+1):
            print("epoch-{}".format(epoch))
            print("-"*200)         
            for j in range(len(batched_training_x)):
                self.forward_propagation(batched_training_x[j])
                self.back_propagation(batched_training_y[j],l2_decay,loss_func,N)
                    
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
            training_accuracy,training_loss= self.compute_loss_accuracy(training_x,training_y,loss_func)            
            val_accuracy,val_loss=self.compute_loss_accuracy(val_x,val_y,loss_func)
            print("=>train_acc={},train_loss={},val_acc={},val_loss={}".format(training_accuracy,training_loss,
                                                                                val_accuracy,val_loss))
            print("-"*200)
            val_acc_hist.append(val_accuracy)
            wandb.log({"epoch":epoch,"train_accuracy":training_accuracy,
                       "train_loss":training_loss,
                      "val_accuracy":val_accuracy,
                      "val_loss": val_loss})
            
            if early_stopping:
                self.update_wei_bias_history(patience,weights_history,bias_history)
                if self.do_early_stopping(patience,prev_val_loss,val_loss,incr_val_loss_count,weights_history,bias_history):
                    print("Early stopping event has occured!!!")
                    print("val_accuracy (before the event): ", val_acc_hist[int(-1*(patience+1))])
                    break
                prev_val_loss=val_loss
            
            batched_training_x,batched_training_y=self.shuffle_batches(batched_training_x,batched_training_y,seed)
                            
    def training(self,x,y,val_percent=0.1,loss_func="cross_entropy",optimizer="GD",
                 max_epoch=10,batch_size=1,learning_rate=0.02,beta=0.9,rmsbeta=0.9,
                 beta_1=0.99,beta_2=0.995,epsilon=10**-8,L2_decay=0.005,
                 early_stopping=False,patience=3,data_augmentation=False,seed=0):
        
        x,y=self.shuffle_dataset(x,y,seed)
        (training_x,training_y),(val_x,val_y)= self.train_val_split(x,y,val_percent)
        if data_augmentation:
            training_x,training_y=self.do_data_augmentation(training_x,training_y)
        N=len(training_x)
        encoded_training_y= self.one_hot_encoding(training_y)
        encoded_val_y=self.one_hot_encoding(val_y)
        
        if optimizer=="GD":
            batch_size=N
        elif optimizer=="SGD":
            batch_size=1
        batched_training_x,batched_training_y= self.segregate_batches(training_x,encoded_training_y,batch_size)
      
        if optimizer=="GD":
            self.gradient_descent(max_epoch,learning_rate,training_x,encoded_training_y,batched_training_x,batched_training_y,val_x,
                                  encoded_val_y,L2_decay,loss_func,early_stopping,patience,seed)
        
        elif optimizer=="SGD":
            self.gradient_descent(max_epoch,learning_rate,training_x,encoded_training_y,batched_training_x,batched_training_y,val_x,
                                  encoded_val_y,L2_decay,loss_func,early_stopping,patience,seed)
            
        elif optimizer=="Mini_batch":
            self.gradient_descent(max_epoch,learning_rate,training_x,encoded_training_y,batched_training_x,batched_training_y,val_x,
                                  encoded_val_y,L2_decay,loss_func,early_stopping,patience,seed)
            
        elif optimizer=="momentumGD":
            self.momentum_gd(max_epoch,learning_rate,beta,training_x,encoded_training_y,batched_training_x,
                             batched_training_y,val_x,encoded_val_y,L2_decay,loss_func,early_stopping,patience,seed)
            
        elif optimizer=="Nesterov":
            self.NAG(max_epoch,learning_rate,beta,training_x,encoded_training_y,batched_training_x,batched_training_y,val_x,
                     encoded_val_y,L2_decay,loss_func,early_stopping,patience,seed)
            
        elif optimizer=="Adagrad":
            self.AdaGrad_RMSprop(max_epoch,learning_rate,rmsbeta,epsilon,training_x,encoded_training_y,batched_training_x,batched_training_y,
                                 val_x,encoded_val_y,L2_decay,loss_func,0,early_stopping,patience,seed)
        
        elif optimizer=="RMSprop":
            self.AdaGrad_RMSProp(max_epoch,learning_rate,rmsbeta,epsilon,training_x,encoded_training_y,batched_training_x,batched_training_y,
                                 val_x,encoded_val_y,L2_decay,loss_func,1,early_stopping,patience,seed)
        elif optimizer=="Adam":
            self.Adam_NAdam(max_epoch,learning_rate,beta_1,beta_2,epsilon,training_x,encoded_training_y,
                            batched_training_x,batched_training_y,val_x,encoded_val_y,L2_decay,loss_func,0,early_stopping,patience,seed)
        elif optimizer=="NAdam":
            self.Adam_NAdam(max_epoch,learning_rate,beta_1,beta_2,epsilon,training_x,encoded_training_y,
                            batched_training_x,batched_training_y,val_x,encoded_val_y,L2_decay,loss_func,1,early_stopping,patience,seed)
            
    def predict(self,data_x,encoded=True):
        num_of_classes = self.last_layer_neuron_count
        L=self.layers_count
        
        self.forward_propagation(data_x)
        y_predicted=np.argmax(self.h[L],axis=1)
        if encoded:
            y_predicted=self.one_hot_encoding(y_predicted)
        return y_predicted