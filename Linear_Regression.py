import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from Regression import BaseRegression

class MyLinearRegression(BaseRegression):

    def _hypothesis(self):
        return np.dot(self.X,self.theta)

    def _gradient(self):
        return (1/self.X.shape[0])*self.X.T.dot(self.X.dot(self.theta)-self.y)

    def _cost(self):
        h=self._hypothesis()
        J = np.dot((h - self.y).T, (h - self.y))
        return J/self.X.shape[0]

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

    def _s_gradient(self,index,random_id):
        index=random_id[index]
        xi=self.X[index,:]
        yi=self.y[index]
        return (xi*(np.dot(xi,self.theta)-yi)).reshape(self.theta.shape[0],1)

    def _sto_get_params(self):
        """
        Get parameter theta using Stochastic gradient descent
        """
        count=0
        last_checked=self.theta
        for it in range(10):
            random_id=np.random.permutation(self.X.shape[0])
            for index in range(self.X.shape[0]):
                count+=1
                self.theta=self.theta-self.learning_rate*self._s_gradient(index,random_id)
                if count%10==0:
                    this_check=self.theta
                    if np.linalg.norm(this_check-last_checked)/len(last_checked)<10**-3:
                        #checking if the difference is still significant, if not, just return theta.
                        print(self._cost())
                        return self.theta
                    last_checked=this_check
        print(self._cost())
        return self.theta

    def fit(self,X,y):
        """
        Automatically use Stochastic Gradient Descent for large datasets, Batch Gradient Descent for small datasets
        """
        one = np.ones((X.shape[0], 1))
        self.X = np.concatenate((one, X), axis=1)
        self.y = y
        self.theta = np.zeros((self.X.shape[1], 1))
        if self.X.shape[0]>10000: return self._sto_get_params()
        return self._get_params()



if __name__=="__main__":
    X = np.random.rand(20000,1)
    y = 4 + 3 * X + .5*np.random.randn(20000, 1)

    plt.plot(X,y,'ro')
    test=MyLinearRegression(0.1,100,0.1)
    theta=test.fit(X,y)
    print(theta)
    test_x=np.linspace(0,1)
    test_y=theta[0]+theta[1]*test_x
    plt.plot(test_x,test_y)
    plt.show()
