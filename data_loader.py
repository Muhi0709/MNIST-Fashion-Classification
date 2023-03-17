import numpy as np
from keras.datasets import fashion_mnist,mnist

def load_fashion_mnist(scaling="MinMax"):
    r=np.random.default_rng(seed=100000)
    (X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()
    num_of_examples=X_train.shape[0]
    l=list(range(num_of_examples))
    r.shuffle(l)
    train_x=np.empty((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    train_y=np.empty(y_train.shape)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
    for i in range(num_of_examples):
        train_x[i]=X_train[l[i]].flatten()
        train_y[i]=y_train[l[i]]

    if scaling=="standard_normal":
        train_x=(train_x - np.mean(train_x,axis=0))/np.std(train_x,axis=0)
        X_test=(X_test - np.mean(X_test,axis=0))/np.std(X_test,axis=0)
        
    elif scaling=="MinMax":
        train_x=(train_x - np.min(train_x,axis=0))/(np.max(train_x,axis=0)-np.min(train_x,axis=0))
        X_test=(X_test - np.min(X_test,axis=0))/(np.max(X_test,axis=0)-np.min(X_test,axis=0))

    return (train_x,train_y),(X_test,y_test)

def load_mnist(scaling="MinMax"):
    r=np.random.default_rng(seed=100001)
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    num_of_examples=X_train.shape[0]
    l=list(range(num_of_examples))
    r.shuffle(l)
    train_x=np.empty((X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
    train_y=np.empty(y_train.shape)
    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
    for i in range(num_of_examples):
        train_x[i]=X_train[l[i]].flatten()
        train_y[i]=y_train[l[i]]

    if scaling=="standard_normal":
        train_x=(train_x - np.mean(train_x,axis=0))/np.std(train_x,axis=0)
        X_test=(X_test - np.mean(X_test,axis=0))/np.std(X_test,axis=0)
        
    elif scaling=="MinMax":
        train_x=(train_x - np.min(train_x,axis=0))/(np.max(train_x,axis=0)-np.min(train_x,axis=0))
        X_test=(X_test - np.min(X_test,axis=0))/(np.max(X_test,axis=0)-np.min(X_test,axis=0))

    return (train_x,train_y),(X_test,y_test)
