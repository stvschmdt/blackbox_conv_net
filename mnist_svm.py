"""
classifier program for recognizing handwritten digits from the MNIST
data set, using an SVM classifier"""

import mnist_loader 
from sklearn import svm
from sklearn.externals import joblib
from logger import Logging

class Simple_SVM():
    def __init__(self):
        self.training_data, self.validation_data, self.test_data = mnist_loader.load_data()
        #print training_data[0][0].shape, training_data[0][0].shape
        # train
        self.clf = svm.SVC()
        self.clf.fit(self.training_data[0][:1000], self.training_data[1][:1000])
        # test
        self.predictions = [int(a) for a in self.clf.predict(self.test_data[0])]
        self.true_positives = [ (self.test_data[0][idx],p[0],p[1]) for idx,p in enumerate(zip(self.predictions, self.test_data[1])) if p[0]==p[1] ]
        num_correct = sum(int(a == y) for a, y in zip(self.predictions, self.test_data[1]))
        #print 'simple black box classifier using an SVM'
        logger.info('black box svm accuray: %f' % (float(num_correct)/float(len(self.test_data[1]))))
        joblib.dump(self.clf, '/home/ubuntu/machinelearning/src/nicefolk/simple_svm.pkl')

if __name__ == "__main__":
    bb = Simple_SVM()
    #print bb.true_positives[:1]

    
