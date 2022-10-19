from random import sample
import numpy as np
import scipy.spatial.distance as dist
import time as time
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Question1(object):
    def bayesClassifier(self,data,pi,means,cov):
        logPriors = np.log(pi)
        term1 = means.dot(np.linalg.inv(cov)).dot(data.T)
        term2 = -1/2 * means.dot(np.linalg.inv(cov)) * (means) 
        term2 = logPriors + term2.sum(1)   
        obj = term2.reshape(3,1) + term1
        return np.argmax(obj, axis= 0)

    def classifierError(self,truelabels,estimatedlabels):
        error = np.sum(np.absolute(estimatedlabels - truelabels) > 0)/truelabels.size
        return error


class Question2(object):
    def trainLDA(self,trainfeat,trainlabel):
        nlabels = int(trainlabel.max())+1 # Assuming all labels up to nlabels exist. 
        pi = np.zeros(nlabels)            # Store your prior in here # 3 by 0
        means = np.zeros((nlabels,trainfeat.shape[1]))            # Store the class means in here  # 3 by 2
        cov = np.zeros((trainfeat.shape[1],trainfeat.shape[1]))   # Store the covariance matrix in here # 3 by 3
        # In your implementation, all quantities should be ordered according to the label.
        # This means that your pi[i] and means[i,:] should correspond to label class i.
        # Put your code below

        for i in trainlabel:
            if i == 0:
                pi[0] += 1
            elif i == 1:
                pi[1] += 1
            else:
                pi[2] += 1
        pi /= trainlabel.size

        for i in range(nlabels):
            means[i] = np.mean(trainfeat[trainlabel == i], axis=0)


        for i in range(nlabels):
            for a in range(trainlabel.size):
                if trainlabel[a] == i:
                    cov += np.dot((trainfeat[a] - means[i]).reshape(2,1), (trainfeat[a] - means[i]).reshape((1,2)))

        cov /= (trainlabel.size - nlabels)



        # Don't change the output!
        return (pi,means,cov)

    def estTrainingLabelsAndError(self,trainingdata,traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        sampledata = self.trainLDA(trainingdata, traininglabels)
        esttrlabels = q1.bayesClassifier(trainingdata,sampledata[0], sampledata[1],sampledata[2])
        trerror = q1.classifierError(traininglabels, esttrlabels)
        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self,trainingdata,traininglabels,valdata,vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)
        sampledata = self.trainLDA(trainingdata, traininglabels) # I assume i dont use valdata to train it
        estvallabels = q1.bayesClassifier(valdata,sampledata[0], sampledata[1],sampledata[2])
        valerror = q1.classifierError(vallabels, estvallabels)

        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self,trainfeat,trainlabel,testfeat, k):
        temp = dist.cdist(trainfeat,testfeat, 'euclidean')
        # print(temp.shape)
        sortedlabels = np.zeros((k, testfeat.shape[0]))
        temp1 = np.argpartition(temp, k, axis = 0)
        for i in range(k):
            for j in range(testfeat.shape[0]):
                sortedlabels[i][j] = trainlabel[temp1[i][j]]
        
        # print(sortedlabels.shape)
        # print(sortedlabels)
        
        estimatedlabels = stats.mode(sortedlabels, axis = 0)[0]
        # print(estimatedlabels.shape)
        # print(estimatedlabels)
        return estimatedlabels.reshape(testfeat.shape[0])
    

    def kNN_errors(self,trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1,3,4,5]

        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            k = k_array[i]
            esttrlabels = self.kNN(trainingdata, traininglabels, trainingdata, k)
            estvallabels = self.kNN(trainingdata, traininglabels, valdata, k)
            trainingError[i] = q1.classifierError(traininglabels, esttrlabels)
            validationError[i] = q1.classifierError(vallabels, estvallabels)
        
        
        # Don't change the output!
        return (trainingError, validationError)

class Question4(object):
    def sklearn_kNN(self,traindata,trainlabels,valdata,vallabels):
        classifier, valerror, fitTime, predTime = (None, None, None, None)
        classifier = neighbors.KNeighborsClassifier(n_neighbors= 1, algorithm= 'brute' )
        t1 = time.time()
        classifier.fit(traindata,trainlabels)
        fitTime =  time.time() - t1
        
        t1 = time.time()
        estlabels = classifier.predict(valdata)
        predTime = time.time()- t1
        q1 = Question1()
        valerror = q1.classifierError(vallabels, estlabels)

        # Don't change the output!

        
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self,traindata,trainlabels,valdata,vallabels):
        classifier, valerror, fitTime, predTime = (None, None, None, None)
        classifier = LinearDiscriminantAnalysis()

        t1 = time.time()
        classifier.fit(traindata,trainlabels)
        fitTime =  time.time() - t1
        
        t1 = time.time()
        estlabels = classifier.predict(valdata)
        predTime = time.time()- t1
        q1 = Question1()
        valerror = q1.classifierError(vallabels, estlabels)
        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
