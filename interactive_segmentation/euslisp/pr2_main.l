#!/usr/bin/env roseus

(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "interactive_segmentation")

(require "package://pr2eus/pr2-interface.l")

(if (not
     (boundp '*pr2*))
    (pr2-init))
(if (not
     (boundp '*irtviewer*))
    (make-irtviewer))
(send *irtviewer* :objects *pr2*)

(ros::ros-info "RUNNING")



(defun boundingbox-array-callback(msg)
  (setq target-box (car (send msg :boxes)))
  (setq dimension (send target-box :dimensions))
  (setq pose (send target-box :pose))

  
  
  (setq x (send (send target-box :dimensions) :x))
  (print x)
  (ros::ros-warn "subscribing")
  )


(defun main()
  (do-until-key
   (ros::spin-once))
  )

(main)

(ros::subscribe "/cluster_decomposer_final/boxes" jsk_recognition_msgs::BoundingBoxArray #'boundingbox-array-callback)
