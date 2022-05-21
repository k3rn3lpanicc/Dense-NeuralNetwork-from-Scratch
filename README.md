# NN-from-scratch
Simple Dense Neural Network implementation with Numpy in python

```python
ST=[X.shape[0],100,100,100,Y.shape[0]] #It is the network map , you can change the layers (add or remove) and also the number of units in each layer , but don't mess with first (input) and last(outpu) layer
#Todo : Remember to change the Activation function list if you changed number of layers!
lg=["Relu","Relu","Relu","Softmax"] #here you can specify the Activation function for all layers . you can use Relu,Softmax,Tanh,LeakyRelu and sigmoid
```

### Train Process
```python
backpropagation() # does a single backpropagation and updates the weights, do it [epoch-num] times
```
### Save Trained Model
```python
savemodel([Best_W,Best_b,acuracy(),best_cost,ST,lg,lr,landa],"fileName.data")#saves weights and hyperparameters to a file with pickle library
```
### Load Model (in Model_predict.py file):
```python
W,b,acurate,cosst,ST,lg,lr,landa=loaddt("fileName.data") #reads hyperparameters and network weights from saved file
```

### Predict the result
Just do a single feedforward on X's data and get the result
```python
result = feed(Xs)
```

### usage
I have trained a black and white picture classification between rubber and sharpener and the accuracy went to '100.0%' and got loss less that '1e-6'