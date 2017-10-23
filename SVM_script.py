import io
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.cross_validation import cross_val_score


#reading the training sample vectors from the csv files and putting it in a panda data struct
print 'reading training data...'
train_df = pd.read_csv('training_vector_probs.csv')
y_df = train_df['class']
del train_df['class']
del train_df['ID']
X = train_df.values

#just to check my X matrix size and Y matrix
print 'X matrix dimensions:'
print np.shape(X)
#sparsifying my X matrix which contains all training sample vectors
sX = sparse.csr_matrix(X)
print 'Y array dimensions: '
Y = y_df.values.transpose()
print np.shape(Y)

#X is going to be an array of size [samples, # of chars]
#y is an array of class labels of size [samples]

# #initialize the classifier
clf = svm.LinearSVC()
# #fit model into the classifier
print 'fitting model...'
clf.fit(sX,Y)


#cross validating purposes
# print 'Cross Validating...'
# c_validation = cross_val_score(clf, sX, Y, scoring='accuracy')
# print c_validation.mean()

#reading test data vectors.. and putting it in the panda data struct
print 'reading test data...'
test = pd.read_csv('test_vector_probs.csv')
del test['ID']
X_test = test.values
#sparsifying the matrix
sX_test = sparse.csr_matrix(X_test)
#predicting for each test sample
print 'initializing prediction sequence...'
pred = clf.predict(sX_test)
print 'prediction starting...'
output = io.open('SVM_predictions.csv', 'w', encoding='utf-8')
count = 0
#writing all prediction into a csv file
for x in np.nditer(pred):
    output.write(str(count) + u',' + str(x) + u'\n')
    count += 1
    print count
output.close()
