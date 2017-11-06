import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt

def plotData(X, y):
    m, dim = np.shape(X)
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'yo')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.show()

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def costFunction(theta, X, y):
    m, dim = np.shape(X)
    #theta = np.reshape(theta, (dim,1))
    htheta = sigmoid(np.dot(X, theta))
    J = (-1.0/m)*(y*np.log(htheta) + (1-y)*(np.log(1-htheta))).sum()
    return J

def gradient(theta, X, y):
    htheta = sigmoid(np.dot(X, theta))
    grad = (1.0/m)*(np.dot(np.transpose(X), (htheta - y)))
    #grad = np.ndarray.flatten(grad)
    return grad

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    for iter in range(iterations):
        grad = gradient(theta, X, y)
        theta = theta - alpha*grad
        print costFunction(theta, X, y)
    return theta

##Loading the data
data = np.loadtxt('ex2data1.txt', delimiter=',')
m, dim = np.shape(data)
X = data[:,0:dim-1]
y = data[:,[dim-1]]

##Plotting the data
#plotData(X, y)

X = np.hstack((np.ones((m,1)), X))
theta = np.zeros((dim,1))

J = costFunction(theta, X, y)

theta = gradientDescent(X, y, theta, 0.01, 10000000)

myargs = (X, y)
#scopt.fmin_bfgs(costFunction, x0=theta, args=myargs)
