import numpy as np
import matplotlib.pyplot as plt

def featureNormalize(X):
    m = np.shape(X)[0]
    dim = np.shape(X)[1]
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X-mu)/sigma
    return X, mu, sigma

def computeCostMulti(X, y, theta):
    m = np.shape(X)[0]
    J = (((np.dot(X,theta) - y)**2).sum())/(2*m)
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        theta = theta - (alpha/m)*np.dot(np.transpose(X),(np.dot(X,theta) - y))
        J_history[iter] = computeCostMulti(X, y, theta)
    return theta, J_history

def normalEqn(X, y):
    Xt = np.transpose(X)
    theta = np.dot(np.dot(np.linalg.inv(np.dot(Xt, X)),Xt),y)
    return theta

##Load the data
data = np.loadtxt('ex1data2.txt', delimiter = ',')
X = data[:,0:2]
y = data[:,[2]]
m = np.shape(X)[0]
dim = np.shape(X)[1]

##Normalize the training data
X, mu, sigma = featureNormalize(X)

##Adding the bias to the training data
X = np.hstack((np.ones((m,1)),X))

##Running gradient descent
alpha = 0.01
num_iters = 400

##Initialize weights
theta = np.ones((dim+1,1))

theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)

plt.plot(np.arange(num_iters),J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
#plt.show()

##Estimate the price of a 1650 sq-ft, 3 br house
x = [1650, 3]
x = (x-mu)/sigma
x = np.hstack((1,x))
price = np.dot(x, theta)
price = ''.join(map(str,price))
print 'Predicted price of 1650 sq feet and 3 br house using gradient descent is', price

##Solving with normal equations

data = np.loadtxt('ex1data2.txt', delimiter = ',')
X = data[:,0:2]
y = data[:,[2]]
m = np.shape(X)[0]
dim = np.shape(X)[1]

X = np.hstack((np.ones((m,1)),X))

theta = normalEqn(X, y)

##Estimate the price of a 1650 sq-ft, 3 br house
x = [1650, 3]
x = np.hstack((1,x))
price = np.dot(x, theta)
price = ''.join(map(str,price))
print 'Predicted price of 1650 sq feet and 3 br house using normal equation is', price
