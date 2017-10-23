# COMP551_A2
Language Classifier


This Package is built for language classification of 5 languages (0 = Slovak, 1 = French, 2 = Spanish,3 = German, 4 = Polish) on a dataset of short utterances provided by training_set_x.csv and training_set_y.csv. 

training_set_x.csv has the Id number of an utterance in column 1 and the utterance (String of characters) in column 2
training_set_y.csv has the Id number of the utterance in column 1 and its language (0,1,2,3,4) in column 2
The Id numbers between the two correspond

Additionally there is a testing data set contained in the file test_set_x.csv which has no y values associated with it. 


Before running the package the following packages need to be installed using pip
      pip install pandas
      pip install numpy
      pip install scikit
      
Now for running the classification algorithms:
1) Run charcountsgenerator.py 
          python charcountsgenerator.py
          
2) Create probability vectors of your training set and testing set instances with the following python scripts
          python trainsetvectors.py
          python testsetvectors.py
   These turn your training and testing instances into vectors of length 614 (all characters in the dataset) where each value in the vector is the probability of a given character
   
   
3) Run one of the classification algorithms
     K-nearest-neighbors: python KNN.py 
        -runs a k-nearest neighbors with default value k = 17 (but can be changed by going into the code)
        -17 was optimized by cross-validation 
        -Additionally it has a default weight function on the instances of inverse distance
        -It also contains a cross-validation class which can be run from inside 
        -The algorithm takes a long time to run and we suggest running overnight
     
     Brute Force - Naive Bayes - SVM ensemble: python bestensemble.py
        -This algorithm combines the three algorithms mentioned above
        -It first checks if we have already seen the instance in our training set
        -Then it runs Naive Bayes to detect unique characters for a given language
        -Then finally it classifies remaining instances based by running SVM only on latin characters
        -This algorithm had the highest classification accuracy on our test set
     
     Support Vector Machine: python SVM_script.py
        -Runs a linear SVM on the dataset
        -SVM from the scikit learn package
     
     Linear Discriminant Analysis: python LDA.py
        -LDA run from the sckit learn package
     
     AdaBoost on Decision Trees: python AdaBoost.py
        -AdaBoost is run on a default number of 200 decision trees from the scikit learn package
       
     
     
    
