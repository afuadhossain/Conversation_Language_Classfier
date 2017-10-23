import io
import pandas as pd
import numpy as np

#KNN class initalizes a KNN classifier
class KNN:
    #inialize KNN with training_data X and classes y along with numbr of neighbors
    def __init__(self, k, X, y, weight='distance'):
        self.k = k
        self.weight = weight
        self.X = X
        self.y = y

    #predict makes a prediction on a test set
    #returns a list of predictions
    def predict(self, X_test):
        predictions = []
        for x in range(0,X_test.shape[0]):
            current = Classify(self, X_test[x,:])
            predictions.append(current.ClassifyTestSample(self.k))

        return predictions



class Classify:
    #classify takes in a knn classifier and a test set
    def __init__(self, knn, x_test):
        self.distances = []
        for x in range(0, knn.X.shape[0]):
            self.distances.append(Compare(knn.X[x,:], x_test, knn.y[x]))

    #run quickselect to get the k smallest elements of array
    #quickselect is a variant of quick sort that only puts the k smallest elements
    # in the first k elements of the array. NOt necessarily in order
    def getKSmallest(self,k, l, r):
        pos = self.partition(l,r)

        if (pos-l == k-1):
            return
        if (pos-l > k-1):
            return self.getKSmallest(k, l, pos-1)
        return self.getKSmallest(k-pos+l-1, pos+1, r)

    def swap(self, x, y):
        temp = self.distances[x]
        self.distances[x] = self.distances[y]
        self.distances[y] = temp


    def partition(self, l, r):
        pivot = self.distances[r]

        i = l
        for x in range(l, r):
            if (self.distances[x].compareTo(pivot) == -1):
                self.swap(i, x)
                i+=1
        self.swap(i, r)
        return i

    #get class returns the class of a given instance
    def getClass(self,k):
        arr = np.zeros(5)
        currentMax = 0.0
        maxClass = 0
        for x in range(0,k):
            arr[self.distances[x].category] = 1.0/self.distances[x].distance
            if arr[self.distances[x].category] > currentMax:
                currentMax = arr[self.distances[x].category]
                maxClass = self.distances[x].category

        return maxClass

    def ClassifyTestSample(self, k):
        self.getKSmallest(k, 0, len(self.distances)- 1)
        return self.getClass(k)


#compare class stores the euclidean distance between x_train and x_test as well as the class of x_train
class Compare:
    def __init__(self, x_train, x_test, category):
        self.distance = np.linalg.norm(x_train-x_test)
        self.category = category
    #we compare to Compare classes based on their euclidean distance
    def compareTo(self, c2):
        if self.distance < c2.distance:
            return -1
        else:
            return 1

class CrossValidate:
    def __init__(self, X, y, k):
        self.X = [] #X is a list of k numpy arrays
        self.y = [] #y is a list of k numpy vectors
        a = len(y)/k #a = 1/k * (size of training set)
        for x in range(0,k-1):
            #divide training set into a list of K equal parts
            self.X.append( X[(a*x):(a*(x+1)), :])
            self.y.append(y[(a*x):((x+1)*a)])

        self.X.append(X[4*x:X.shape[0],:])
        self.y.append(y[4*x:X.shape[0]])

        self.k = k

    def cross_validate(self,neighbors):

        accuracy = np.zeros(5)
        #iterate through the list of numpy arrays and run cross validation
        #i indicates the set which is being held out on each iteration
        for i in range(0, self.k):
            if i == 0:
                #train stores 4/5 of training set to train KNN model
                train = self.X[1]
                #y_train stores 4/5 of classes to train KNN model
                #y_train and train values correspond perfectly
                y_train = self.y[1]
                for j in range(2, self.k):
                    if j == i: #don't include one subset
                        continue
                    train = np.concatenate([train,self.X[j]], axis= 0)
                    y_train = np.concatenate([y_train, self.y[j]], axis=1)
                knn = KNN(neighbors, train,y_train)
                pred = knn.predict(self.X[i])
                accuracy[i] = self.get_accuracy(i, pred)
            else:
                train = self.X[0]
                y_train = self.y[0]
                for j in range(1, self.k):
                    if j == i: #don't include one subset
                        continue
                    train = np.concatenate([train,self.X[j]], axis= 0)
                    y_train = np.concatenate([y_train, self.y[j]], axis=1)
                knn = KNN(17, train,y_train)
                pred = knn.predict(self.X[i])
                accuracy[i] = self.get_accuracy(i, pred)

        return accuracy.mean() #take the average of all the cross validations



    def get_accuracy(self, k, predictions):
        #predictions is the predicted values on the set of cross validation
        count = 0 #count is the total number of correct predictions
        for x in range(0,len(predictions)):
            if predictions[x] == self.y[k][x]: #if predictions is correct increase count
                count+= 1
        return float(count)/float(len(predictions)) #return the accuracy


#read training data into a numpy array
def readTrainingData(filename):
    train_df = pd.read_csv(filename)
    y_df = train_df['class']
    del train_df['class']
    del train_df['ID']

    X = train_df.values #convert from dataframe to numpy array
    Y = y_df.values.transpose()
    return X,Y

#reads the test data into a numpy array
def readTestData(filename):
    test = pd.read_csv(filename)
    del test['ID']
    X_test = test.values
    return X_test


if __name__ == '__main__':
    X,y = readTrainingData("training_vector_probs.csv")
    X_test = readTestData("test_vector_probs.csv")
    '''
    To run cross validation you can run the following
        cv = CrossValidate(X,y,5)
        print cv.cross_validate(17)
    Would recommend iterating thorugh values of k

    '''
    pred = knn.predict(X_test[0:20, :])

    output = io.open('KNN_predictions.csv','w', encoding = 'utf-8')

    output.write(u'Id,category\n')
    for pos,x in enumerate(pred):
        output.write(u''+str(pos) + u',' + str(x) + u'\n')

    output.close()
