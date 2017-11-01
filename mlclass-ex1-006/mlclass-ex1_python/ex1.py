import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y):
    plt.plot(X, y, 'rx')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.show()

def computeCost(X, y, theta):
    m = len(y)
    J = ((np.dot(X,theta) - y)**2).sum()/(2*m)
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y)
    for iter in range(iterations):
        theta = theta - alpha*(1.0/m)*np.dot(np.transpose(X),(np.dot(X, theta) - y))
    return theta

##Load data from the text file
data = np.loadtxt('ex1data1.txt', delimiter = ',')
X = data[:,[0]]
y = data[:,[1]]
m = len(y)                              #Number of training samples

##Plot the data
plotData(X, y)

##Running gradient descent
X = np.hstack((np.ones((m,1)), X))      #add bias
theta = np.zeros((2,1))                 #initial weights

iterations = 1500
alpha = 0.01

J = computeCost(X, y, theta)
theta = gradientDescent(X, y, theta, alpha, iterations)

plt.plot(data[:,[0]], np.dot(X, theta))
plotData(data[:,[0]], y)

##Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(([1, 3.5]), theta)
predict1  = ''.join(map(str,predict1*10000))
print 'For population of 35,000, we predict a profit of',predict1
predict2 = np.dot(([1, 7]), theta)
predict2  = ''.join(map(str,predict2*10000))
print'For population = 70,000, we predict a profit of',predict2
