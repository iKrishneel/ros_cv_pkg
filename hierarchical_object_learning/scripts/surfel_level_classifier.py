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

def load_trained_model_manifest(filename):
    return joblib.load('../.ros/' + filename)

def object_classifier_predict(fvect, model_fname):
    clf = None
    try:
        clf = load_trained_model_manifest(model_fname)
    except ValueError as err:
        print (err.args)
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    fvect = imp.fit_transform(fvect)                               
    #response = clf.predict_proba(fvect)
    response = clf.predict(fvect)
    return response
        
def train_object_surfel_classifier(feature_vector, label_vector, filename):
    try:
        imp = Imputer(missing_values='NaN', strategy='median', axis=1)
        feature_vector = imp.fit_transform(feature_vector)
        clf = SVC(C=100, cache_size=200, class_weight=None,
                  coef0=0.0, degree=3, gamma=0.0, kernel='linear',
                  max_iter=-1, probability=True,
                  random_state=None, shrinking=True, tol=0.00001, verbose=False)
        clf.fit(feature_vector, label_vector)
        joblib.dump(clf, filename + '.pkl')
    except ValueError as err:
        print (err.args)
        
def convert_from_feature_list(feature_list):
    rows = len(feature_list)
    cols = len(feature_list[0].histogram)
    feature_vector = np.zeros((rows, cols), dtype=np.float)
    for i in range(rows):
        for j in range(cols):
            feature_vector[i,j] = feature_list[i].histogram[j]
    return feature_vector

def surfel_level_classifier_handler(req):
    feature_vector = convert_from_feature_list(req.features.feature_list)
    save_fname = str(req.model_save_path)
    if req.run_type is 0:
        print "TRAINING CLASSIFIER"
        label_vector = req.features.labels
        train_object_surfel_classifier(feature_vector, label_vector, save_fname)
        return PredictorResponse([], 1)
    elif req.run_type is 1:
        print "RUNNING CLASSIFIER"
        response = object_classifier_predict(feature_vector, save_fname)
        return PredictorResponse(response, 1)
    else:
        print "\033[31mERROR: THE SURFEL LEVEL RUN_TYPE IS NOT SET.\033[0m"
    
def object_classifier_predictor_server():
    rospy.init_node('surfel_level_predictor_server')
    s = rospy.Service('surfel_level_classifier',
                      FitFeatureModel, surfel_level_classifier_handler)
    rospy.spin()
    
def onInit():
    object_classifier_predictor_server()
    
if __name__ == "__main__":
    onInit()
