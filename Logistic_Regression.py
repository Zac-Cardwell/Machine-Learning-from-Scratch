'''
Logistic regression is a supervised machine learning algorithm mainly used for classification tasks 
where the goal is to predict the probability that an instance of belonging to a given class.it’s referred 
to as regression because it takes the output of the linear regression function as input and uses a sigmoid 
function to estimate the probability for the given class.
'''

'''
The linear part of the model (the weighted sum of the inputs) calculates the log-odds of a successful event, 
specifically, the log-odds that a sample belongs to class 1.

    log-odds = bias + w1x1 + w2x2 + ....+ wmxm

Given the probability of success (p) predicted by the logistic regression model, 
we can convert it to odds of success as the probability of success divided by the probability of not success

The log-odds of success can be converted back into an odds of success by calculating the exponential of the log-odds.

    odds = exp(bias + w1x1 + w2x2 + ....+ wmxm)

Finally, the odds of success can be converted into the probability  of sucesses.

    p = odds/(1+odds) or 1/(1+exp(log-odds))
'''
            
''' 
For this project, i will be using the Maximum Likelihood Estimation (MLE) in order to estimate the paramaters of the model.
The goal is to maximize the conditional probability  of observing the data X given the probability  distribution  theta

    P(x:theta) 

here x is the joint probability distribution of all observations in the problem domain. the resulting probability can be referred
to as the likelihood function denoted as L(). 

In order to reduce Multiplying many small probabilities together, the problem is often rewritten as the sum of the log conditional probability. 
And, since its common to try and reduce the cost function instead of maximizeing it,  the negative of the log-likelihood function is used,
referred to as a Negative Log-Likelihood (NLL) function

    -sum i to m log(P(xi:theta))
'''

'''
In order to use maximum likelihood, we need to first find the probobility distribution. In the case of logistic regression, a  binomial probability 
distribution is assumed for the data sample, where each example is one outcome of a Bernoulli trial that takes the probability of a successful outcome. 

    P(y=1) = p
    P(y=0) = 1-p

The expected value of the Bernoulli distribution can be calculated as:
    
    mean = p*1 + (1-p)*0

When implemented in the model it will look like:
    
    mean = yhat * y + (1-yhat) * (1-y)
    
This function will always return a large probability when the model is close to the matching class value, and a small value when it is far away

    minimize sum i to n -(log(yhat_i) * y_i + log(1 – yhat_i) * (1 – y_i))
'''


import numpy as np
import matplotlib.pyplot as plt

 
class regression:
    def __init__(self, input_size):
        self.weight = np.zeros(input_size)
        self.bias = 1


    def sigmoid(self, z):
        out = 1/(1+np.exp(-z))
        return out
    
    def MLE(self, y, yhat):
        answer = 0
        for i in range(len(y)):
            if y[i] == 1:
                answer -= np.log(yhat[i])
            elif y[i] == 0:
                answer -= np.log(1-yhat[i])
            else:
                continue
        return answer
            

    def train(self, x, y, epoch, lr):
        temp=0
        
        for _ in range(epoch):
            
            temp = np.dot(x, self.weight) + self.bias
            yhat = self.sigmoid(temp)
            
            if _%10 == 0:
                print(self.MLE(y, yhat))
            
            
            self.weight += lr * np.dot(x.T, (y-yhat))
            self.bias += lr * sum((y-yhat))
            
    def plot(self, x, y):
        # plot the data and separating line
        plt.scatter(x[:,0], x[:,1], c=y, s=100, alpha=0.5)
        x_axis = np.linspace(-6, 6, 100)
        y_axis = -(self.weight[0] + x_axis*self.weight[1]) / self.bias
        plt.plot(x_axis, y_axis)
        plt.show()
            
            
N = 100
D = 2

X = np.random.randn(N,D)

# center the first 50 points at (-2,-2)
X[:50,:] = X[:50,:] - 2*np.ones((50,D))

# center the last 50 points at (2, 2)
X[50:,:] = X[50:,:] + 2*np.ones((50,D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)
            
            
model = regression(2) 
model.train(X, T, 100, .1)    
model.plot(X, T)       
            

            