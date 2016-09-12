# Sinhalese Sentiment Analyzer
A Simple Sinhalese sentiment anaylzer using the SVM algorithm from scikit learn

Note : This application is still in a primitive stage and does have various flaws which will be addressed incrementally in future versions.
The testing data is fed from the test.txt file, and the training data set in fed from the trainData directory which consists of sub directories
for each of its classification classes

#### Dependencies
    -python 2.7
    -numpy 1.11.1
    -scipy 0.18.0
    -scikit-learn 0.17.1

#### Installation 

```sh
$ pip install sklearn 
```
#### Application Usage

```sh
#run the default tests
$ python main.py
#run a specific file with tests
$ python main.py -f <filename>
#run a specific test from command line
$ python main.py -s <sentence>
```