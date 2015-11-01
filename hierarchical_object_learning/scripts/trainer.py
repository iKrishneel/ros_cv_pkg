#!/usr/bin/env python

import numpy as np

import rospy
import roslib
roslib.load_manifest("hierarchical_object_learning")

# from object_recognition.srv import Trainer
# from object_recognition.srv import TrainerResponse

from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import Imputer

from hierarchical_object_learning.srv import FitFeatureModel
from hierarchical_object_learning.srv import FitFeatureModelResponse

def object_classifier_train(feature_vector, label_vector, path):
    #np.savetxt('/home/krishneel/Desktop/first.txt', feature_vector)
    
    imp = Imputer(missing_values='NaN', strategy='median', axis=1)
    feature_vector = imp.fit_transform(feature_vector)

    #np.savetxt('/home/krishneel/Desktop/second.txt', feature_vector)

    
    clf = SVC(C=100, cache_size=200, class_weight=None,
              coef0=0.0, degree=3, gamma=0.0, kernel='linear',
              max_iter=-1, probability=True,
              random_state=None, shrinking=True, tol=0.00001, verbose=False)
    clf.fit(feature_vector, label_vector)
    joblib.dump(clf, path + '.pkl')

    
def fit_feature_model_handler(req):
    path = req.model_save_path
    feature_vector = np.array(req.features.feature_list)  # is a 2D array
    label_vector = np.array(req.features.labels)
    #feature_vector = feature_vector.reshape(-1, stride)
    feature_vector = feature_vector.astype("float")
    label_vector = label_vector.astype("float")
    
    if feature_vector.shape[0] == label_vector.shape[0]:
        object_classifier_train(feature_vector, label_vector, path)
        return TrainerResponse(1)
    else:
        return TrainerResponse(0)

def fit_feature_model_server():
    rospy.init_node('fit_feature_model_server')
    s = rospy.Service('fit_feature_model',
                      FitFeatureModel, fit_feature_model_handler)
    rospy.spin()

if __name__ == "__main__":
    fit_feature_model_server()
