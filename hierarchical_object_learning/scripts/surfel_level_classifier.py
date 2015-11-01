#!/usr/bin/env python

import numpy as np

import rospy
import roslib
roslib.load_manifest("hierarchical_object_learning")

from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

from hierarchical_object_learning.srv import Classifier
from hierarchical_object_learning.srv import ClassifierResponse

clf = None

def load_trained_model_manifest():
    return joblib.load('../.ros/svm.pkl')

def object_classifier_predict(fvect):
    try:
        imp = Imputer(missing_values='NaN', strategy='median', axis=1)
        fvect = imp.fit_transform(fvect)                               
        #response = clf.predict_proba(fvect)
        response = clf.predict(fvect)
        return response
    except ValueError as err:
        print (err.args)
    
def surfel_level_predictor_handler(req):
    feature_vector = np.array(req.feature)
    feature_vector = feature_vector.astype("float")
    
    response = object_classifier_predict(fvect)
    return PredictorResponse(response)
    
def object_classifier_predictor_server():
    rospy.init_node('surfel_level_predictor_server')
    s = rospy.Service('surfel_level_predictor',
                      Classifier, surfel_level_predictor_handler)
    rospy.spin()

    
def onInit():
    global clf
    clf = load_trained_model_manifest()
    object_classifier_predictor_server()
    
if __name__ == "__main__":
    onInit()
