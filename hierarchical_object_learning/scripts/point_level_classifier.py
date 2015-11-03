#!/usr/bin/env python

import numpy as np

import rospy
import roslib
roslib.load_manifest("hierarchical_object_learning")

from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

from hierarchical_object_learning.srv import FitFeatureModel
from hierarchical_object_learning.srv import FitFeatureModelResponse

model_extension = '.pkl'

def load_trained_model_manifest(filename):
    return joblib.load(filename + model_extension)

def object_classifier_predict(feature_list, model_name):
    gaussian_naive_bayes = None
    try:
        gaussian_naive_bayes = load_trained_model_manifest(model_name)
    except ValueError as err:
        print (err.args)
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    feature_list = imp.fit_transform(feature_list)
    responses = gaussian_naive_bayes.predict(feature_list)
    return responses

def train_object_point_classifier(feature_vector, label_vector, filename):
    try:
        imp = Imputer(missing_values='NaN', strategy='median', axis=1)
        feature_vector = imp.fit_transform(feature_vector)
        gaussian_naive_bayes = GaussianNB()
        gaussian_naive_bayes.fit(feature_vector, label_vector)
        joblib.dump(gaussian_naive_bayes, filename + model_extension, compress=3)
    except ValueError as err:
        print(err.args)

def convert_from_feature_list(feature_list):
    rows = len(feature_list)
    cols = len(feature_list[0].histogram)
    feature_vector = np.zeros((rows, cols), dtype=np.float)
    for i in range(rows):
        for j in range(cols):
            feature_vector[i,j] = feature_list[i].histogram[j]
    return feature_vector

def object_level_classifier_handler(req):
    feature_vector = convert_from_feature_list(req.features.feature_list)
    save_fname = str(req.model_save_path)    
    if req.run_type is 0:
        print "TRAINING POINT CLASSIFIER"
        label_vector = np.array(req.features.labels)        
        train_object_point_classifier(feature_vector, label_vector, save_fname)
        return FitFeatureModelResponse([], 1)
    elif req.run_type is 1:
        print "RUNNING POINT LEVEL CLASSIFIER"
        response = object_classifier_predict(feature_vector, save_fname)
        return FitFeatureModelResponse(response, 1)
    else:
        print "\033[31mERROR: THE POINT LEVEL RUN_TYPE IS NOT SET.\033[0m"
    
def object_classifier_predictor_server():
    rospy.init_node('point_level_predictor_server')
    s = rospy.Service('point_level_classifier',
                      FitFeatureModel, object_level_classifier_handler)
    rospy.spin()

def onInit():
    global gaussian_naive_bayes
    gaussian_naive_bayes = load_trained_model_manifest()
    object_classifier_predictor_server()
    
if __name__ == "__main__":
    onInit()
