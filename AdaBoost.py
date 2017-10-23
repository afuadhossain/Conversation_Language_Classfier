import io
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import AdaBoostClassifier

counts = io.open('charcounts.csv', 'r', encoding='utf-8')

train_df = pd.read_csv('KNN_probs.csv')
y_df = train_df['class']
del train_df['class']
del train_df['ID']

X = train_df.values
Y = y_df.values.transpose()



clf = AdaBoostClassifier(n_estimators=400)
clf.fit(X,Y)

test = pd.read_csv('test_vector_probs.csv')
del test['ID']
X_test = test.values

print X_test

pred = clf.predict(X_test)

output = io.open('AdaBoost_400.csv', 'w', encoding='utf-8')
count = 0
output.write(u'Id,category\n')
for x in np.nditer(pred):
    output.write(str(count) + u',' + str(x) + u'\n')
    count += 1
output.close()
