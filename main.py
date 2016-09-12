#!/usr/bin/env python
# coding: utf-8
import sys
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':

    # training data set directory
    data_dir = './trainData'

    train_data = []
    train_labels = []
    test_data = []
    #binary classification classes
    classes = ['pos','neg']

    for current in classes:
        dirname = os.path.join(data_dir, current)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:               
                content = f.read()
                #if file is starting with pos append to training data as a positive statement
                if fname.startswith('pos'):
                    train_data.append(content)
                    train_labels.append('positive')
                # else it belongs to negative category
                else :
                    train_data.append(content)
                    train_labels.append('negative')

    #CountVectorizer will find the number of occurences of a word in the test data.
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(train_data)
    vocabulary = count_vectorizer.vocabulary_

    #print the vocabulary
    print "\n\n===== Vocabulary ===="
    for key in vocabulary:
        print key, " : ", vocabulary[key] 
    print "===== END: Vocabulary ====\n\n"
    
    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=1,
                                 max_df = 0.5,
                                 sublinear_tf=True,
                                 use_idf=True,decode_error='ignore')

    train_vectors = vectorizer.fit_transform(train_data)
    classifier_rbf = svm.SVC()
    classifier_rbf.fit(train_vectors, train_labels)

    #read from test.txt file
    with open('test.txt','r') as testData:
        for line in testData:
            test_data.append(line);  
            print " ==========start============= "
            print line
            print classifier_rbf.predict(vectorizer.transform(test_data)) 
            print " ==========end============= \n"
            
            test_data = []



    