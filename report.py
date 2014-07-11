from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from pybrain.structure import FeedForwardNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer


boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target)

X_train, y_train = X, y
X_est=[[11.95,0.00,18.100,0,0.6590,5.6090,90.00,1.385,24,680.0,20.20,332.09,12.13]]

####### PART I ########
### 1 AB
print "Adaptive Boosting results"
y_est=zeros(10)
for k in range(0,10):
    regressor = AdaBoostRegressor(n_estimators=60)
    regressor.fit(X_train, y_train)
    y_est[k]=regressor.predict(X_est)
    print y_est[k]
y_est_exp=mean(y_est)
print "the avg est is: ",y_est_exp

### 2 DT
print "Decision Tree results"
y_est=zeros(10)
for k in range(0,10):
    regressor =DecisionTreeRegressor(max_depth=7)
    regressor.fit(X_train, y_train)
    y_est[k]=regressor.predict(X_est)
    print y_est[k]
y_est_exp=mean(y_est)
print "the avg est is: ",y_est_exp

####### PART III ########
### 1 AB
print "Adaptive Boosting results"
y_est_all=[]
for num in range(50,100,10):
    y_est=zeros(10)
    for k in range(0,10):
        regressor = AdaBoostRegressor(n_estimators=num)
        regressor.fit(X_train, y_train)
        y_est[k]=regressor.predict(X_est)
    y_est_exp=mean(y_est)
    print "the avg est of ",num," learners is: ",y_est_exp
    y_est_all.append(y_est_exp)
y_est_all_exp=mean(y_est_all)
y_est_all_var=var(y_est_all)
print "mean: ",y_est_all_exp,"; variance: ",y_est_all_var 


### 2 DT
print "Decision Tree results"
y_est_all=[]
for num in range(5,10):
    y_est=zeros(10)
    for k in range(0,10):
        regressor = DecisionTreeRegressor(max_depth=num)
        regressor.fit(X_train, y_train)
        y_est[k]=regressor.predict(X_est)
    y_est_exp=mean(y_est)
    print "the avg est of ",num," max depth is: ",y_est_exp
    y_est_all.append(y_est_exp)
y_est_all_exp=mean(y_est_all)
y_est_all_var=var(y_est_all)
print "mean: ",y_est_all_exp,"; variance: ",y_est_all_var 


### 3 kNN
print "k Nearest Neighbors results"
y_est_all=[]
for num in range(3,8):
    y_est=zeros(10)
    for k in range(0,10):
        neigh = KNeighborsRegressor(n_neighbors=num) 
        neigh.fit(X_train, y_train)
        y_est[k]=neigh.predict(X_est)
    y_est_exp=mean(y_est)
    print "the avg est of ",num," nearest neighbors is: ",y_est_exp
    y_est_all.append(y_est_exp)
y_est_all_exp=mean(y_est_all)
y_est_all_var=var(y_est_all)
print "mean: ",y_est_all_exp,"; variance: ",y_est_all_var 


### 4 NN
print "Neural Network results"
net = []
net.append(buildNetwork(13,7,1))
net.append(buildNetwork(13,9,5,1))
net.append(buildNetwork(13,9,7,3,1))
net.append(buildNetwork(13,9,7,3,2,1))
net_arr = range(0, len(net))

max_epochs = 50
train_err = zeros(len(net))
ds = SupervisedDataSet(13, 1)
for j in range(1, len(X_train)):
    ds.addSample(X_train[j], y_train[j])

y_est_all=[]
for i in range(0, len(net)):
    trainer = BackpropTrainer(net[i], ds)
    for k in range(1, max_epochs):
        train_err[i] = trainer.train()
    y_est= net[i].activate(X_est[0]) 
    print "the est of ",i+1," hidden layers is: ",y_est
    y_est_all.append(y_est)
y_est_all_exp=mean(y_est_all)
y_est_all_var=var(y_est_all)
print "mean: ",y_est_all_exp,"; variance: ",y_est_all_var 




