import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

class BaseRegression:
    def __init__(self,learning_rate=0.01,max_iter=100,gamma=0):
        """
        :param learning_rate: the name explains itself
        :param max_iter: the name explains itself
        :param gamma: this param is used for Gradient Descent with momentum
        How well the model works depends greatly on how you choose the param, especially learning rate
        """
        self.learning_rate=learning_rate
        self.max_iter=max_iter
        self.gamma=gamma

    def predict(self,X_test):
        one = np.ones((X_test.shape[0], 1))
        X_test=np.concatenate((one,X_test),axis=1)
        return np.dot(X_test,self.theta)

    def score(self,X_test,y_test):
        y_pred=self.predict(X_test)
        return r2_score(y_test,y_pred)