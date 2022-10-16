import numpy as np

class NaiveBayesSpamFilter:
    def __init__(self):
        self.pSpamArray=np.NaN
        self.pHamArray=np.NaN
        self.pSpam_=np.NaN
    
    def train(self,X_train,y_train):
        n = X_train.shape[0]
        m = X_train.shape[1]
        self.pSpam= sum(y_train) / float(n)
        
        spamArray = np.ones(m)
        hamArray =  np.ones(m)
        
        numberSpams = 2.0
        numberHams = 2.0
        
        for i in range(0, n):
            if y_train[i] == 1:
                spamArray += X_train[i]
                numberSpams += sum(X_train[i]) 
            else:
                hamArray += X_train[i]
                numberHams += sum(X_train[i])
                
        self.pSpamArray = np.log(spamArray / numberSpams)
        self.pHamArray = np.log(hamArray / numberHams)
    
    def classify(self,x):
        p0 = sum(x * self.pHamArray) + np.log(1 - self.pSpam)
        p1 = sum(x * self.pSpamArray) + np.log(self.pSpam)
        if p0 > p1:
            return 0
        else:
            return 1
    
    def testSuccess(self,X_test,y_test):
        error = 0
        for i in range (0,X_test.shape[0]):
            x = X_test[i]
            smsType_aral=self.classify(x)
            if smsType_aral != y_test[i]:
                error += 1
        print 'Success rate of filter: %', (1000-error) / 10.0