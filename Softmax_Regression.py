import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics import accuracy_score
from Regression import BaseRegression

class MySoftmaxRegression(BaseRegression):
    def _hypothesis(self,Z):
        exp=np.exp(Z-np.max(Z,axis=0,keepdims=True))
        return exp/exp.sum(axis=0)

    def _gradient(self):
        return self.X.T.dot((self._hypothesis(np.dot(self.X,self.theta))-self.y.T))

    def _get_params(self):
        """
        This is inefficient (really don't know why)
        """
        v_old=np.zeros_like(self.theta)
        for it in range(self.max_iter):
            v_new=self.gamma*v_old+self.learning_rate*self._gradient()
            self.theta=self.theta-v_new
            if np.linalg.norm(self._gradient()) < 10 ** -3:
                break
            v_old=v_new
        return self.theta

    def _sto_get_params(self):
        class_nums=self.theta.shape[1]
        last_checked=self.theta
        count=0
        while count<self.max_iter:
            random_id = np.random.permutation(self.X.shape[1])
            for index in random_id:
                xi=self.X[index,:].reshape(self.X.shape[1],1)
                yi=self.y[:,index].reshape(class_nums,1)
                ai=self._hypothesis(np.dot(self.theta.T,xi))
                self.theta=self.theta+self.learning_rate*np.dot(xi,(yi-ai).T)
                count+=1
                if count%20==0:
                    if np.linalg.norm(self.theta-last_checked)<10**-3:
                        return self.theta
        return self.theta

    def fit(self,X,y):
        self.X=np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        y=y.reshape(y.shape[0])
        C=len(np.unique(y))
        self.y=sparse.coo_matrix((np.ones_like(y),
        (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
        self.theta=np.zeros((self.X.shape[1],C))
        return self._sto_get_params()

    def predict(self,X_test):
        X_test=np.concatenate((np.ones((X_test.shape[0],1)),X_test),axis=1)
        return np.argmax(self._hypothesis(np.dot(X_test,self.theta)),axis=1)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)

if __name__=="__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris=load_iris()
    X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target)
    y_train=y_train.reshape(y_train.shape[0])
    y_test=y_test.reshape(y_test.shape[0])
    model=MySoftmaxRegression(0.01,10000)
    model.fit(X_train,y_train)
    print("Score on the iris datasets (test):",model.score(X_test,y_test))
    print("Score on the iris datasets (train):",model.score(X_train,y_train))


