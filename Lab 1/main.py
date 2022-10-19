from copy import deepcopy
import numpy as np


class Lab1(object):
    def solver(self, A, b):

        return np.linalg.solve(A, b)

    def fitting(self, x, y):
        # coeff = np.zeros(2)
        a = np.column_stack((x, np.ones(len(x))))
        a = np.linalg.pinv(a)
        return np.dot(a,y)


    def naive5(self, X, A, Y):
        # Calculate the matrix with $(i,j$)-th entry as  $\mathbf{x}_i^\top A \mathbf{y}_j$ by looping over the rows of $X,Y$.
        
        Sx = np.zeros((X.shape[0], Y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                Sx[i,j]=np.dot(np.dot(X[i],A),Y[j])

        return Sx

    def matrix5(self, X, A, Y):
        # Repeat part (a), but using only matrix operations (no loops!).

        return np.dot(np.dot(X,A),Y.T)

    def naive6(self, X, A):
        # Calculate a vector with $i$-th component $\mathbf{x}_i^\top A \mathbf{x}_i$ by looping over the rows of $X$.
        Sx = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            Sx[i]=np.dot(np.dot(X[i],A),X[i])
        return Sx
    def matrix6(self, X, A):
        # Repeat part (a) using matrix operations (no loops!).
        return np.sum(np.dot(X,A)*X,axis=1)
