#!/usr/bin/env python
# coding: utf-8
import sys
import os
import pybst.avltree as pyavl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from pybst.draw import plot_tree

if __name__ == '__main__':

    # training data set directory
    data_dir = './trainData'

    train_data = []
    train_labels = []
    test_data = []
    stop_words_data = []
    #binary classification classes
    classes = ['pos','neg', 'stopwords']

    for current in classes:
        dirname = os.path.join(data_dir, current)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:               
                content = f.read()
                #if file is starting with pos append to training data as a positive statement
                if fname.startswith('pos'):
                    train_data.append(content)
                    train_labels.append('positive')
                #if it belongs to negative category
                elif fname.startswith('neg') :
                    train_data.append(content)
                    train_labels.append('negative')
                else:
                    stop_words_data.append(content)

    #CountVectorizer will find the number of occurences of a word in the test data.
    count_vectorizer = CountVectorizer()
    count_vectorizer.fit_transform(train_data)
    vocabulary = count_vectorizer.vocabulary_

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=1,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 stop_words=stop_words_data,
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

    #tokenize sentence to words
    def visualizer(sentence):
        result = [[],[]]
        words = sentence.split()
        for word in words:
            temp = [word]
            buildTree(temp, result)
        plotTree(result)
    
    def buildTree (data, result):   
        for line in data:
            test_data.append(line);
            decision = classifier_rbf.decision_function(vectorizer.transform(test_data))[0]
            prediction = classifier_rbf.predict(vectorizer.transform(test_data))[0]
            if prediction == 'positive' and decision != 0:
                result[0].append((line.strip('\n'), decision))
            elif prediction == 'negative' and decision != 0:
                result[1].append((line.strip('\n'), decision))
            del test_data[:]

    #plots and displays the tree
    def plotTree(nodes):
        tree = pyavl.AVLTree()
        for node in nodes:
            for nod in node:
                tree.insert(nod[1], node[0])
        plot_tree(tree)

    #execute the analyser
    def execute(data):
        for index, line in enumerate(data):
            test_data.append(line)
            print '{}) {} : {}'.format(index + 1, line.strip('\n'), classifier_rbf.predict(vectorizer.transform(test_data))[0])
            del test_data[:]

    if len(sys.argv) == 1:
        readFromFile()
    elif len(sys.argv) == 3 and sys.argv[1] == '-f':
        readFromFile(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == '-s':
        readFromSentence(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == '-v':
        visualizer(sys.argv[2])