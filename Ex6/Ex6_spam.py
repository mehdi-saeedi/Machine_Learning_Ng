# spam classifier
# Ex6 of Andrew Ng's course in Machine Learning (part 2)

import numpy as np
from scipy.io import loadmat
from sklearn import svm
import re
import string
from nltk.stem import PorterStemmer
import pandas as pd


# get vocabulary list
def getVocabList():
    return pd.read_csv('vocab.txt', sep='\t', header=None).values[:, 1]


# process a given email by changing it to a canonical format
def processEmail(email_contents):
    vocabList = np.array(getVocabList())
    email_contents = email_contents.lower()

    email_contents = re.sub('<[^<>]+>', ' ', email_contents)  # html tag
    email_contents = re.sub('[0-9]+', 'number', email_contents)  # number
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)  # utl
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)  # email address
    email_contents = re.sub('[$]+', 'dollar', email_contents)  # dollar sign

    translator = str.maketrans('', '', string.punctuation)
    email_contents = email_contents.translate(translator)
    stemmer = PorterStemmer()

    word_indices = []
    wordList = re.sub("[^\w]", " ", email_contents).split()

    for word in wordList:
        word = re.sub('[^0-9a-zA-Z]', '', word)
        word = stemmer.stem(word)

        idx = np.where(vocabList == word)
        if len(idx[0]) != 0:
            word_indices.append(idx[0][0] + 1)

    return vocabList.shape[0], word_indices


# Extract features
def emailFeatures(n, word_indices):
    x = np.zeros((n, 1))
    x[word_indices] = 1
    return x


vocabList = np.array(getVocabList())

# sample test
spam_train = loadmat('spamTrain.mat')
X = spam_train['X']
y = spam_train['y']
model_linear = svm.SVC(kernel='linear', C=1)
model_linear.fit(X, y.flatten())
p = model_linear.predict(X).reshape(-1, 1)
print("Training accuracy = ", np.mean(np.array(p == y).astype(int)) * 100)

# sample test
spam_test = loadmat('spamTest.mat')
Xtest = spam_test['Xtest']
ytest = spam_test['ytest']
p = model_linear.predict(Xtest).reshape(-1, 1)
print("Evaluating the trained Linear SVM on a test set = ", np.mean(np.array(p == ytest).astype(int)) * 100)

weight = np.sort(model_linear.coef_)
idx = np.argsort(model_linear.coef_)
print('\nTop predictors of spam: \n')
for i in range(0, 14):
    j = weight.shape[1] - i - 1
    print("vocab = ", vocabList[idx[0, j]])
    print("weight =", weight[0, j])

filenames = ['emailSample1.txt', 'emailSample2.txt', 'emailSample3.txt', 'spamSample1.txt', 'spamSample2.txt',
             'spamSample3.txt']

# Test
for filename in filenames:
    file_contents = open(filename, 'r').read()
    nWord, word_indices = processEmail(file_contents)
    features = emailFeatures(nWord, word_indices).reshape(1, -1)
    p = model_linear.predict(features)
    if p == 1:
        result = 'Yes'
    else:
        result = 'No'
    print(filename, 'is ', result)

print('done')
