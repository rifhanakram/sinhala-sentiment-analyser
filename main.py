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
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True,decode_error='ignore')

    train_vectors = vectorizer.fit_transform(train_data)
    classifier_rbf = svm.SVC()
    classifier_rbf.fit(train_vectors, train_labels)

    #read from a given file or the default file
    def readFromFile(file='test.txt'):
        with open(file,'r') as testData:
            execute(testData)
    
    #read from a command line argument
    def readFromSentence(sentence):
        temp = [sentence]
        execute(temp)

    #execute the analyser
    def execute(data):
        for index, line in enumerate(data):
            test_data.append(line);  
            print '{}) {} : {}'.format(index + 1, line.strip('\n'), classifier_rbf.predict(vectorizer.transform(test_data))[0])
            del test_data[:]

    if len(sys.argv) == 1:
        readFromFile()
    elif len(sys.argv) == 3 and sys.argv[1] == '-f':
        readFromFile(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == '-s':
        readFromSentence(sys.argv[2])