import io
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

counts = io.open('charcounts.csv', 'r', encoding='utf-8')

train_df = pd.read_csv('training_vector_probs.csv')
y_df = train_df['class']
del train_df['class']
del train_df['ID']

X = train_df.values
Y = y_df.values.transpose()


sklearn_lda = LinearDiscriminantAnalysis()
sklearn_lda.fit(X,Y)



test = pd.read_csv('test_vector_probs.csv')
del test['ID']
X_test = test.values

print X_test

pred = sklearn_lda.predict(X_test)

output = io.open('3NN_Ytest.csv', 'w', encoding='utf-8')
count = 0
for x in np.nditer(pred):
    output.write(str(count) + u',' + str(x) + u'\n')
    count += 1
output.close()
