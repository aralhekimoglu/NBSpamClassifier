from NaiveBayesSpamFilter import NaiveBayesSpamFilter
from ParsingUtils import importData

dataDirectory = './data/SMSSpamCollection.txt'
testCount=1000

print "Importing and Cleaning Data..."
X,y= importData(dataDirectory)

X_train=X[testCount:]
y_train=y[testCount:]
X_test=X[0:testCount]
y_test=y[0:testCount]

print "Training Filter..."
nbfilter=NaiveBayesSpamFilter()
nbfilter.train(X_train,y_train)
print "Testing..."
nbfilter.testSuccess(X_test,y_test)