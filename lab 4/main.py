from cProfile import label
from re import L
import numpy as np
from sklearn import neighbors
import scipy.spatial.distance as dist
from sklearn import linear_model
from sklearn.model_selection import train_test_split

class Question1(object):
    def kMeans(self,data,K,niter):
        """ Implement the K-Means algorithm.

        **For grading purposes only:**

        Do NOT change the random seed, otherwise we are not able to grade your code! This is true throughout this script. However, in practice, you never want to set a random seed like this.
        For your own interest, after you have finished implementing this function, you can change the seed to different values and check your results.
        Please use numpy library for initial random choice. This will then use the prescribed seed. Scipy library is using a different seeding system, so that would probably result in an error during our grading.

        Parameters:
        1. data     (N, d) numpy ndarray. The unlabelled data with each row as a feature vector.
        2. K        Integer. It indicates the number of clusters.
        3. niter    Integer. It gives the number of iterations. An iteration includes an assignment process and an update process.

        Outputs:
        1. labels   (N,) numpy array. It contains which cluster (0,...,K-1) a feature vector is in. It should be the (niter+1)-th assignment.
        2. centers  (K, d) numpy ndarray. The i-th row should contain the i-th center.
        """
        N = data.shape[0]
        d =data.shape[1]
        np.random.seed(12312)   # Do NOT change the seed!
        centers = data[np.random.choice(data.shape[0], K, replace=False),:]
        # Put your code below
        clusters = np.zeros(N) 
        counter = 0
        while(counter < niter):
            clusters = np.zeros(N) 
            distmat = dist.cdist(data, centers)
            # print(clusters.shape)
            # print(distmat.shape)
            for i in range(N):
                clusters[i] = int(np.argmin(distmat[i]))
            for a in range(K):
                centers[a] = np.average(axis= 0, a= data[clusters == a,:])
            counter +=1

        labels = clusters
            
        # Remember to check your data types: labels should be integers!
        return (labels.astype(int), centers)

    def calculateJ(self,data,kMeansfun):
        """ Calculate the J_k value for K=2,...,10.

        This function should call the passed kMeansfun() function and set niter=100.

        Parameters:
        1. data         (N, d) numpy ndarray. The unlabelled data with each row as a feature vector.
        2. kMeansfun    Correct kMeans function. You should use kMeansfun wherever you want to use kMeans in function calculateJ. This is used for grading purposes.

        Outputs:
        1. err          (9,) numpy array. The i-th element contains the J_k value when k = i+2.
        """
        err = np.zeros(9)
        # Put your code below
        for a in range(err.size):
            labels, centers = kMeansfun(data, a+2, 100)
            temp = 0
            for i in range(data.shape[0]):
                # print(centers.shape)
                # print(data[i] - centers[round(labels[i]),:])
                temp += np.sqrt(np.sum(element**2 for element in (data[i] - centers[round(labels[i]),:])))
            err[a] = temp
        print(err)
        return err

from sklearn.cluster import KMeans

class Question2(object):
    def trainVQ(self,image,B,K):
        """ Generate a codebook for vector quantization.

        Please use the KMeans function from the sklearn package. You can use kmeans.cluster_centers_ to get the cluster centers after you fit your model.
        Also, ensure that your threadpoolctl is at least of version 3.

        For grading purposes only: Please flatten any matrix in *row-major* order. You can use np.flatten(xxx) to flatten your matrix.

        Parameters:
        1. image        (N, M) numpy ndarray. It represents a grayscale image.
        2. B            Integer. You will use B×B blocks for vector quantization. You may assume that both N and M are divisible by B.
        3. K            Integer. It gives the size of your codebook.

        Outputs:
        1. codebook     (K, B^2) numpy ndarray. It is the codebook you should return.
        2. kmeans       KMeans Object. For grading only.
        """
        np.random.seed(12345)
        # Put your code below
        a = np.split(ary =image, indices_or_sections = image.shape[0]/B, axis = 0)
        a = np.array(a)
        temp = np.split(ary= a,indices_or_sections = image.shape[1]/B, axis = 2)
        temp = np.array(temp)
        temp = temp.reshape((temp.shape[1] *temp.shape[0], 5,5))
        temp = temp.reshape(-1, 25)
        kmeans = KMeans(n_clusters= K)
        kmeans.fit(temp)
        codebook = kmeans.cluster_centers_
        # print(codebook.shape)
        return (codebook,kmeans)

    def compressImg(self,image,codebook,B):
        """ Compress an image using a given codebook.

        You can use the nearest neighbor classifier from scikit-learn if you want (though it is not necessary) to map blocks to their nearest codeword.

        **For grading purposes only:**

        Please flatten any matrix in *row-major* order. You can use np.flatten(xxx) to flatten your matrix.

        Parameters:
        1. image        (N, M) numpy ndarray. It represents a grayscale image. You may assume that both N and M are divisible by B.
        2. codebook     (K, B^2) numpy ndarray. The codebook used in compression.
        3. B            Integer. Block size.

        Outputs:
        1. cmpimg       (N//B, M//B) numpy ndarray. It consists of the indices in the codebook used to approximate the image.
        """
        # Put your code below
        cmpimg = np.zeros((image.shape[0]// B, image.shape[1]//B))
        # cmpimg = np.zeros((image.size//(B**2)))
        a = np.split(ary =image, indices_or_sections = image.shape[0]/B, axis = 0)
        a = np.array(a)
        temp = np.split(ary= a,indices_or_sections = image.shape[1]/B, axis = 2)
        temp = np.array(temp)
        temp = temp.reshape((temp.shape[1] *temp.shape[0], 5,5))
        temp = temp.reshape(-1, 25)
        for i in range(image.size//(B**2)):
        #Check that your indices are integers!
            diff = codebook - temp[i]
            diff = np.sum(diff * diff, axis= 1)
            cmpimg[i%(image.shape[0]// B)][i// (image.shape[0]// B)] = round(np.argmin(diff))
        # print(cmpimg)
        return cmpimg.astype(int)

    def decompressImg(self,indices,codebook,B):
        """ Reconstruct an image from its codebook.

        You can use np.reshape() to reshape the flattened array.

        Parameters:
        1. indices      (N//B, M//B) numpy ndarray. It contains the indices of the codebook for each block.
        2. codebook     (K, B^2) numpy ndarray. The codebook used in compression.
        3. B            Integer. Block size.

        Outputs:
        1. rctimage     (N, M) numpy ndarray. It consists of the indices in the codebook used to approximate the image.
        """
        # Put your code below
        rctimage = np.zeros((indices.shape[0]* B, indices.shape[1]* B))

        for a in range(indices.shape[0]):
            for b in range(indices.shape[1]):
                rctimage[(B*a):(B*a)+B,(B*b):(B*b)+B] = codebook[indices[a][b]].reshape(B,B)
                
        return rctimage

class Question3(object):
    def generatePrototypes(self,traindata,trainlabels,K_list):
        """ Generate prototypes from labeled data.

        You can use the KMeans function from the sklearn package.

        **For grading purposes only:**

        Do NOT change the random seed, otherwise we are not able to grade your code!

        Parameters:
        1. traindata        (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels      (Nt,) numpy array. The labels in the training set.
        3. K_list           List. A list of integers corresponding to the number of prototypes under each class.

        Outputs:
        1. proto_dat_list   A length len(K_list) list. The K-th element in the list is a (K * num_classes, d) numpy ndarray, representing the prototypes selected if using K prototypes under each class. You should keep the order as in the given K_list.
        2. proto_lab_list   A length len(K_list) list. The K-th element in the list is a (K * num_classes,) numpy array, representing the corresponding labels if using K prototypes under each class. You should keep the order as in the given K_list.
        """
        np.random.seed(56789)   # Do NOT change this seed!
        proto_dat_list = []
        proto_lab_list = []
        # Put your code below
        uniquelab = np.unique(trainlabels)
        for j in K_list:
            tempdata = np.zeros((j* uniquelab.size, traindata.shape[1]))
            templabels = np.zeros((j* uniquelab.size))
            for i in range(uniquelab.size):
                dataofclass = traindata[trainlabels == uniquelab[i],:]
                classifier = KMeans(n_clusters= j)
                classifier.fit(dataofclass)
                tempdata[j* i: j *(i+ 1), :] = classifier.cluster_centers_
                templabels[j * i: j * (i+ 1)] = uniquelab[i]
            proto_dat_list.append(tempdata)
            proto_lab_list.append(templabels.astype(int))
            
        # Check that your proto_lab_list only contains integer arrays!
        return (proto_dat_list, proto_lab_list)

    def protoValError(self,proto_dat_list,proto_lab_list,valdata,vallabels):
        """ Generate prototypes from labeled data.

        You may assume there are at least min(K_list) examples under each class. set(trainlabels) will give you the set of labels.

        Parameters:
        1. proto_dat_list   A list of (K * num_classes, d) numpy ndarray. A list of prototypes selected. This should be one of the outputs from your previous function.
        2. proto_lab_list   A list of (K * num_classes,) numpy array. A list of corresponding labels for the selected prototypes. This should be one of the outputs from your previous function.
        3. valdata          (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels        (Nv,) numpy array. The labels in the validation set.

        Outputs:
        1. proto_err        (len(proto_dat_list),) numpy ndarray. The validation error for each K value (in the same order as the given K_list).
        """
        proto_err = np.zeros(len(proto_dat_list))
        # Put your code below
        for i in range(len(proto_lab_list)):
            currlab = proto_lab_list[i]
            currdata = proto_dat_list[i]
            classifier = neighbors.KNeighborsClassifier(n_neighbors = 1)
            classifier.fit(currdata, currlab)
            proto_err[i] = 1 - classifier.score(valdata, vallabels)

            
        return proto_err

class Question4(object):
    def benchmarkRSS(self,trainfeat,trainresp,valfeat,valresp):
        """ Return the benchmark RSS.

        In particular, always predict the response as zero (mean response on the training data).

        Calculate the validation RSS for this model. Please use the formula as defined in the jupyter notebook.

        Parameters:
        1. trainfeat    (Nt, d) numpy ndarray. The features in the training set.
        2. trainresp    (Nt,) numpy array. The responses in the training set.
        3. valfeat      (Nv, d) numpy ndarray. The features in the validation set.
        4. valresp      (Nv,) numpy array. The responses in the validation set.

        Outputs:
        1. rss          Scalar. The validation RSS.
        """
        # Put your code below
        temp = np.zeros(valresp.size)
        r = valresp - temp
        rss = np.sum(np.dot(r, r.T)) / temp.size
        return rss

    def OLSRSS(self,trainfeat,trainresp,valfeat,valresp):
        """ Return the RSS from the ordinary least squares model.

        Use sklearn.linear_model.LinearRegression() with the default parameters.

        Calculate the validation RSS for this model. Please use the formula as defined in the jupyter notebook.

        Note: The .score() method returns an  R^2 value, not the RSS, so you shouldn't use it anywhere in this problem.

        Parameters:
        1. trainfeat    (Nt, d) numpy ndarray. The features in the training set.
        2. trainresp    (Nt,) numpy array. The responses in the training set.
        3. valfeat      (Nv, d) numpy ndarray. The features in the validation set.
        4. valresp      (Nv,) numpy array. The responses in the validation set.

        Outputs:
        1. rss          Scalar. The validation RSS.
        """
        # Put your code below
        classifier = linear_model.LinearRegression()
        classifier.fit(trainfeat, trainresp)
        yhats = classifier.predict(valfeat)
        r = valresp - yhats
        rss = np.sum(np.dot(r, r.T)) / yhats.size
        return rss

    def RidgeRSS(self,trainfeat,trainresp,valfeat,valresp):
        """ Return the RSS from the ridge regression.

        Apply ridge regression with sklearn.linear_model.Ridge. Sweep the regularization/tuning parameter α = 0,...,100 with 1000 equally spaced values.

        Note: Larger values of α shrink the weights in the model more, and α=0 corresponds to the LS solution.

        Parameters:
        1. trainfeat    (Nt, d) numpy ndarray. The features in the training set.
        2. trainresp    (Nt,) numpy array. The responses in the training set.
        3. valfeat      (Nv, d) numpy ndarray. The features in the validation set.
        4. valresp      (Nv,) numpy array. The responses in the validation set.

        Outputs:
        1. rss_array    (1000,). The validation RSS array. This is used for plotting. This will not be tested by the autograder.
        2. best_a       Scalar. The alpha that minimizes the RSS.
        3. best_rss     Scalar. The corresponding RSS.
        4. coef         (d,) numpy array. The minimizing coefficient. This is for visualization only. This will not be tested by the autograder.
        """
        a = np.linspace(0,100,1000)
        rss_array = np.zeros(a.shape)
        # Put your code below
        for i in range(a.size):
            classifier = linear_model.Ridge(alpha = a[i])
            classifier.fit(trainfeat,trainresp)
            yhats = classifier.predict(valfeat)
            r = valresp - yhats
            rss_array[i] = np.sum(np.dot(r, r.T)) / yhats.size
        
        temp = np.argmin(rss_array)
        best_a = a[temp]
        best_rss = rss_array[temp]
        classifier = linear_model.Ridge(alpha = best_a)
        classifier.fit(trainfeat,trainresp)
        coef = classifier.coef_
        return (rss_array, best_a, best_rss, coef)

    def LassoRSS(self,trainfeat,trainresp,valfeat,valresp):
        """ Return the RSS from the Lasso regression.

        Apply lasso regression with sklearn.linear_model.Lasso. Sweep the regularization/tuning parameter α = 0,...,1 with 1000 equally spaced values.

        Note: Larger values of α will lead to sparser solutions (i.e. less features used in the model), with a sufficiently large value of α leading to a constant prediction. Small values of α are closer to the LS solution, with α=0 being the LS solution.

        Parameters:
        1. trainfeat    (Nt, d) numpy ndarray. The features in the training set.
        2. trainresp    (Nt,) numpy array. The responses in the training set.
        3. valfeat      (Nv, d) numpy ndarray. The features in the validation set.
        4. valresp      (Nv,) numpy array. The responses in the validation set.

        Outputs:
        1. rss_array    (1000,). The validation RSS array. This is used for plotting. This will not be tested by the autograder.
        2. best_a       Scalar. The alpha that minimizes the RSS.
        3. best_rss     Scalar. The corresponding RSS.
        4. coef         (d,) numpy array. The minimizing coefficient. This is for visualization only. This will not be tested by the autograder.
        """
        a = np.linspace(1,0,1000,endpoint=False)[::-1]     # Note: 0 will give an error.
        rss_array = np.zeros(a.shape)
        # Put your code below
        for i in range(a.size):
            classifier = linear_model.Lasso(alpha = a[i])
            classifier.fit(trainfeat,trainresp)
            yhats = classifier.predict(valfeat)
            r = valresp - yhats
            rss_array[i] = np.sum(np.dot(r, r.T)) / yhats.size
        
        temp = np.argmin(rss_array)
        best_a = a[temp]
        best_rss = rss_array[temp]
        classifier = linear_model.Lasso(alpha = best_a)
        classifier.fit(trainfeat,trainresp)
        coef = classifier.coef_
        return (rss_array, best_a, best_rss, coef)
