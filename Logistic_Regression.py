import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Regression import BaseRegression

class MyLogisticRegression(BaseRegression):

    def _hypothesis(self,X):
        return 1/(1+np.exp(-np.dot(X,self.theta)))

    def _gradient(self):
        return (1/self.X.shape[0])*self.X.T.dot(self._hypothesis(self.X)-self.y)

    def _get_params(self):
        """
        Get paramater theta using gradient descent
        set gamma>0 for gradient descent with momentum
        else let it be 0
        """
        v_old=np.zeros_like(self.theta)
        for it in range(self.max_iter):
            v_new=self.gamma*v_old+self.learning_rate*self._gradient()
            self.theta=self.theta-v_new
            if np.linalg.norm(self._gradient())/len(self.theta)<10**-3:
                # checking if the difference is still significant, if not, stop.
                print('break at iter',it)
                print(self._cost())
                break
            v_old=v_new
        else:
            print('break at iter',self.max_iter)
            print(self._cost())
        return self.theta

    def _s_gradient(self, index, random_id):
        index = random_id[index]
        xi = self.X[index, :]
        yi = self.y[index]
        return (xi * (self._hypothesis(xi) - yi)).reshape(self.theta.shape[0], 1)

    def _cost(self):
        return -1*(self.y.T.dot(np.log(self._hypothesis(self.X))))-(1-self.y).T.dot(np.log(1-self._hypothesis(self.X)))

    def _sto_get_params(self):
        """
        Get parameter theta using Stochastic gradient descent
        """
        count = 0
        last_checked = self.theta
        for it in range(10):
            random_id = np.random.permutation(self.X.shape[0])
            for index in range(self.X.shape[0]):
                count += 1
                self.theta = self.theta - self.learning_rate * self._s_gradient(index, random_id)
                if count % 10 == 0:
                    this_check = self.theta
                    if np.linalg.norm(this_check - last_checked) / len(last_checked) < 10 ** -3:
                        # checking if the difference is still significant, if not, just return theta.
                        print(self._cost())
                        return self.theta
                    last_checked = this_check
        print(self._cost())
        return self.theta

    def fit(self,X,y):
        self.X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        self.y=y
        self.theta=np.zeros((self.X.shape[1],1))
        if X.shape[0]>10000: return self._sto_get_params()
        return self._get_params()



if __name__=="__main__":
    X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50,
                   2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]]).reshape(20,1)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]).reshape(20,1)
    test=MyLogisticRegression(1)
    theta=test.fit(X,y)
    print(theta,'theta')
    X1,X2=[],[]
    y1,y2=[],[]
    for index in range(len(X)):
        if y[index]==0:
            X1.append(X[index])
            y1.append(y[index])
        else:
            X2.append(X[index])
            y2.append(y[index])

    plt.plot(X1,y1,'ro')
    plt.plot(X2,y2,'bo')
    line_x=np.linspace(0,6)
    line_y=1/(1+np.exp(-theta[0]-theta[1]*line_x))
    plt.plot(line_x,line_y)
    plt.show()

