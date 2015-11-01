#!/usr/bin/env python

import numpy as np

import rospy
import roslib
roslib.load_manifest("hierarchical_object_learning")

from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

from hierarchical_object_learning.srv import Classifier
from hierarchical_object_learning.srv import ClassifierResponse

gaussian_naive_bayes = None

def load_trained_model_manifest():
    return joblib.load('../.ros/gnb.pkl')

def object_classifier_predict(feature_list):
    try:
        imp = Imputer(missing_values='NaN', strategy='median', axis=1)
        feature_list = imp.fit_transform(feature_list)
        responses = gaussian_naive_bayes.predict(feature_list)
        return responses
    except ValueError as err:
        print (err.args)
    
def object_classifier_predictor_handler(req):

    feature_vector = np.array(req.features.feature_list)
    feature_vector = feature_vector.astype("float")

    response = object_classifier_predict(feature_vector)
    return PredictorResponse(response)
    
def object_classifier_predictor_server():
    rospy.init_node('point_level_predictor_server')
    s = rospy.Service('point_level_predictor',
                      Classifier, object_classifier_predictor_handler)
    rospy.spin()

def onInit():
    global gaussian_naive_bayes
    gaussian_naive_bayes = load_trained_model_manifest()
    object_classifier_predictor_server()
    
if __name__ == "__main__":
    onInit()
