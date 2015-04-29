#!/usr/bin/env python

import numpy
import numpy as np
import rospy

from object_recognition.srv import Trainer
from object_recognition.srv import TrainerResponse
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

def object_classifier_train(fvect, lvect):
    #np.savetxt('/home/krishneel/Desktop/first.txt', fvect)
    
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    fvect = imp.fit_transform(fvect)

    #np.savetxt('/home/krishneel/Desktop/second.txt', fvect)

    
    clf = SVC(C=100, cache_size=200, class_weight=None,
              coef0=0.0, degree=3, gamma=0.0, kernel='linear',
              max_iter=-1, probability=True,
              random_state=None, shrinking=True, tol=0.00001, verbose=False)
    clf.fit(fvect, lvect)
    joblib.dump(clf, '../.ros/svm.pkl')

def object_classifier_trainer_handler(req):
    size = req.size
    stride = req.stride
    f_vec = np.array(req.features)
    lvect = np.array(req.labels)
    fvect = f_vec.reshape(-1, stride)
    fvect = fvect.astype("float")
    lvect = lvect.astype("float")
    
    #print "Printing the Received Vector: \n", lvect.shape
    #print fvect.shape
    
    if fvect.shape[0] == lvect.shape[0]:
        object_classifier_train(fvect, lvect)
        return TrainerResponse(1)
    else:
        return TrainerResponse(0)

def object_classifier_trainer_server():
    rospy.init_node('trainer_server')
    s = rospy.Service('trainer',
                      Trainer, object_classifier_trainer_handler)
    rospy.spin()

if __name__ == "__main__":
    object_classifier_trainer_server()
