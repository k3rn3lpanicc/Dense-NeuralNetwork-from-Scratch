import numpy as np
import random
import readData
import pickle
def loaddt(model):
    with open(model, 'rb') as filehandle:
        variables = pickle.load(filehandle)
        return  variables
W,b,acurate,cosst,ST,lg,lr,landa=loaddt("test_model.data")#use ur model file name here
def g(z,activation="Sigmoid",deriv=False):
    if(activation=="Sigmoid" and deriv==False):
        return 1/(1+np.exp(-z))
    elif (activation=="Relu" and deriv==False):
        return np.maximum(0,z)
    elif(activation=="Sigmoid" and deriv==True):
        return z*(1-z)
    elif(activation=="Relu" and deriv==True):
        z[z<=0] = 0
        z[z>0] = 1
        return z
def feed(Xs):
    global W,Y,lg
    Z=np.dot(W[0],Xs)+b[0]
    A=g(Z,activation=lg[0])
    for i in range(1,len(ST)-1):
        Z=np.dot(W[i],A)+b[i]
        A=g(Z,activation=lg[i])
    return A

#ok for predicting a simple example u can use feed(ur data) to see the results!
