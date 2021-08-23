import cupy as np #Cupy is just a copy of numpy (or more) which works well on GPU , if u can not use that , simply replace cupy with numpy
import random
import read_caltech
import pickle
files1,X_train, y_train = read_caltech.get_data_mat('train') #This one is from another file wich is also in this repository
X=np.array(X_train,dtype=np.float32).T
Y=np.array(y_train,dtype=np.float32).T

'''Hyper Parameters :   '''
lr=0.1 #U can change the learning rate and see the cause in practise
landa=0 #it's for L2 regularization wich i don't use it and it is 0 for that
beta=0.9 #it's for Gradient Descent with momentum
C=Y.shape[0] #it's number of classes , no need to specify it , it will be set automatically from the shape of labels
epsilon=1e-8 #not used yer
ST=[X.shape[0],100,100,100,Y.shape[0]] #It is the network map , you can change the layers (add or remove) and also the number of units in each layer , but don't mess with first (input) and last(outpu) layer
#****Remember to change the Activation function list if you changed number of layers!
lg=["Relu","Relu","Relu","Softmax"] #here you can specify the Activation function for all layers
m=X[0].shape[0] #m is number of examples you have in your dataset (for example 1000 images or something)
Y=Y.reshape(ST[-1],m)
W=[] #weights
b=[] #biases
VdW=[] #it's for momentum
VdB=[] #and this
'''--------------------'''
for i in range(len(ST)-1):
    if(i==0):
        W.append(np.random.randn(ST[i+1],ST[i])*np.sqrt(2/(X.shape[0]))) #i use a method for randomize the weights so that we won't get exploding / vanishing gradients problem
    else:
        W.append(np.random.randn(ST[i+1],ST[i])*np.sqrt(2/(ST[i-1])))
    b.append(np.random.randn(ST[i+1],1)*0.01)
    VdW.append(W[-1]*0)
    VdB.append(b[-1]*0)
def savemodel(dt,filename): #you will see the template of how to use this.
    with open(filename, 'wb') as filehandle:
        pickle.dump(dt, filehandle)
    pass


def g(z, activation="Sigmoid", deriv=False): #it's activation functions , deriv is for derivative and if it's true then it will return the deriv of the function
    if (activation == "Sigmoid" and deriv == False):
        return 1 / (1 + np.exp(-z))
    elif (activation == "Relu" and deriv == False):
        return np.maximum(0, z)
    elif (activation == "LeakyRelu" and deriv == False):#not gonna work well
        return np.maximum(0.01 * z, z)
    elif (activation == "Sigmoid" and deriv == True):
        return z * (1 - z)
    elif (activation == "Relu" and deriv == True):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z
    elif (activation == "LeakyRelu" and deriv == True):
        z[z < 0] = 0.01
        z[z >= 0] = 1
        return z

def feed(Xs=X): #it's our forwardpropagation part for predicting and training
    global W,Y,lg
    Z=np.dot(W[0],Xs)+b[0]
    A=g(Z,activation=lg[0])
    for i in range(1,len(ST)-1 -1):
        Z=np.dot(W[i],A)+b[i]
        A=g(Z,activation=lg[i])
    Z=np.dot(W[-1],A)+b[-1]
    t=np.exp(Z)
    A=t/(np.sum(t,axis=0,keepdims=True))
    return A
def cost(): #it's our cost function that we try to minimize it over time
    ans=0.0
    A=feed()
    global m
    for i in range(m):
        for j in range(C):
            ans+=(-Y[j][i]*np.log(A[j][i]))#(-Y[0][i]*np.log(A[0][i]))+(-(1-Y[0][i])*np.log(1-A[0][i]))#+((landa/(2*m))*np.sum(W,axis=0)) this is used for 2 classes
    return ans


def acuracy(): #acuracy function will return how well we did on train set as a percentage
    A = feed(X).T
    cnt = 0
    yp = Y.T
    for j in range(len(A)):
        # print(np.round(A[j]),yp[j])
        if ((yp[j] - np.round(A[j])).any() == False):
            cnt += 1
    return 100 * (cnt / m)


AS = []


def backpropagation(): #one step of training the NN
    global W, X, Y, b, AS, VdW, VdB
    keep_prob = 0.6
    Z1 = np.dot(W[0], X) + b[0]
    A = g(Z1, activation=lg[0])
    D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob #this line is for dropout , if u don't want to use it just set keep_prob to 1
    A *= D
    A /= keep_prob
    Zs = []
    As = [A]
    for i in range(1, len(ST) - 1 - 1):
        Z = np.dot(W[i], A) + b[i]
        Zs.append(Z)

        A = g(Z, activation=lg[i])
        if (i != len(ST) - 2):
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
        As.append(A)
    Z = np.dot(W[-1], A) + b[-1]
    Zs.append(Z)
    t = np.exp(Z)
    A = t / (np.sum(t, axis=0, keepdims=True))
    As.append(A)
    dAL = 0
    dZL = As[-1] - Y  # np.multiply(dAL,g(As[-1],lg[-1],deriv=True))
    dWL = (1 / m) * np.dot(dZL, As[-2].T) + ((landa / m) * W[-1])
    dBL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
    AS = As
    VdW[-1] = (beta * VdW[-1]) + ((1 - beta) * (dWL)) #momentum part
    VdB[-1] = (beta * VdB[-1]) + ((1 - beta) * (dBL))#this too
    W[-1] -= lr * VdW[-1]
    b[-1] -= lr * VdB[-1]

    for i in range(len(ST) - 1 - 2, -1, -1):
        dAL = np.dot(W[i + 1].T, dZL)
        dZL = np.multiply(dAL, g(As[i], lg[i], deriv=True)) #when deriv is true , we are computing gradients (derivatives)
        if (i == 0):
            dWL = (1 / m) * np.dot(dZL, X.T) + ((landa / m) * W[i])
        else:
            dWL = (1 / m) * np.dot(dZL, As[i - 1].T) + ((landa / m) * W[i])

        dBL = (1 / m) * np.sum(dZL, axis=1, keepdims=True)
        VdW[i] = (beta * VdW[i]) + ((1 - beta) * (dWL))
        VdB[i] = (beta * VdB[i]) + ((1 - beta) * (dBL))
        # print(type(W[i],VdW[i]))
        W[i] -= lr * VdW[i] #now update the weights
        b[i] -= lr * VdB[i]  #and biases

cnt=0
best_cost=cost()
Best_W=W
Best_b=b
import time
epoch_num=1000
start = time.time()
while cnt<epoch_num:
    cnt += 1
    backpropagation()
    if (cost() < best_cost):
        Best_W = W
        Best_b = b
        best_cost = cost()
    if (cnt % 10 == 0):
        print(acuracy(), "Cost : ", cost(), "Time : ", (time.time() - start) * 1000, "Ms")
        start = time.time()
#Now let's save our model
savemodel([Best_W,Best_b,acuracy(),best_cost,ST,lg,lr,landa],"blackandwhite_model.data")
