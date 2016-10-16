#!/usr/bin/env python
# coding: utf-8
import sys
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.chunk.regexp import *

if __name__ == '__main__':

    # training data set directory
    data_dir = './trainData'

    jj_train_data = []
    jj_train_labels = []

    nn_train_data = []
    nn_train_labels = []

    vb_train_data = []
    vb_train_labels = []

    test_data = []

    #binary classification classes
    classes = ['noun', 'adjective']

    for current in classes:
        dirname = os.path.join(data_dir, current)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                #if file is starting with pos append to training data as a positive statement
                if fname.startswith('adjective'):
                    jj_train_data.append(content)
                    jj_train_labels.append('JJ')
                #if it belongs to negative category
                elif fname.startswith('noun') :
                    jj_train_data.append(content)
                    jj_train_labels.append('NN')
                elif fname.startswith('verb') :
                    jj_train_data.append(content)
                    jj_train_labels.append('VB')

    # Create feature vectors
    jj_vectorizer = TfidfVectorizer(min_df=1,
                                 max_df = 0.5,
                                 sublinear_tf=True,
                                 use_idf=True,decode_error='ignore')

    #classifier for adjectives
    jj_train_vectors = jj_vectorizer.fit_transform(jj_train_data)
    jj_classifier_rbf = svm.SVC()
    jj_classifier_rbf.fit(jj_train_vectors, jj_train_labels)

    #read from a given file or the default file
    def readFromFile(file='test.txt'):
        with open(file,'r') as testData:
            execute(testData)

    # #execute the analyser
    def execute(data):
        sentence = []
        result = None
        for index, line in enumerate(data):
            for word in line.split():
                test = [word]
                result_jj = jj_classifier_rbf.predict(jj_vectorizer.transform(test))[0]
                sentence.append((word.decode("utf-8"), result_jj))
                nn_grammar = "NP: {<JJ>?<JJ>*<NN>}"
                parser = RegexpParser(nn_grammar)
                result = parser.parse(sentence)
                del test[:]
        for word in sentence:
            print (word[0] + ": " + word[1])
        print result.draw()

    if len(sys.argv) == 1:
        readFromFile()
    # elif len(sys.argv) == 3 and sys.argv[1] == '-f':
    #     readFromFile(sys.argv[2])
    # elif len(sys.argv) == 3 and sys.argv[1] == '-s':
    #     readFromSentence(sys.argv[2])
    # # elif len(sys.argv) == 3 and sys.argv[1] == '-v':
    # #     #visualizer(sys.argv[2])