#!/usr/bin/env python

import numpy
import numpy as np
import rospy

from object_recognition.srv import Predictor
from object_recognition.srv import PredictorResponse

from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

 #later change this to load once
def object_classifier_loader():
    return joblib.load('../.ros/svm.pkl')

def object_classifier_predict(fvect):
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    fvect = imp.fit_transform(fvect)                               
    clf = object_classifier_loader()
    #response = clf.predict_proba(fvect)
    response = clf.predict(fvect)
    #return response[0,0]
    return response
    
def object_classifier_predictor_handler(req):
    stride = req.stride
    f_vec = np.array(req.feature)
    fvect = f_vec.reshape(-1, stride)
    fvect = fvect.astype("float")
    
    response = object_classifier_predict(fvect)
    #print response
    return PredictorResponse(response)
    #else:
    #return PredictorResponse(0)
    
def object_classifier_predictor_server():
    rospy.init_node('predictor_server')
    s = rospy.Service('predictor',
                      Predictor, object_classifier_predictor_handler)
    rospy.spin()

if __name__ == "__main__":
    object_classifier_predictor_server()
