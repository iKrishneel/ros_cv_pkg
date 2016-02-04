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

(setq *manip-pose* #f(131.783 54.9158 -1.3071 0.272882 -88.7191 -357.653 -5.85497 360.042 -45.3354 1.53671 2.39065 -95.778 -1.10987 -5.93392 359.975 1.67125 48.912))

(setq *init-pose* (float-vector 12.0 45.0 60.0 0.0 -120.0 0.0 -45.0 0.0 -45.0 60.0 0.0 -120.0 0.0 -45.0 0.0 0.0 35.0))

(load "generate-tabletop-scene.l")

(ros::ros-info "RUNNING")

(defun manip-pose(&optional(time 5000))
  (send *pr2* :angle-vector *manip-pose*)
  (if (boundp '*ri*)
      (send *ri* :angle-vector (send *pr2* :angle-vector) time)
    (send *ri* :wait-interpolation)
    )
  )

(defun send-target-pose-to-robot(pose-vector &optional (arm :larm) &optional (time 5000))
  (when (boundp '*ri*)
    (send *ri* :angle-vector (send *pr2* :angle-vector) time)
    (send *ri* :wait-interpolation)
    )
  )

(defun check-arm-reachability(pose-vector &optional (arm :larm))
  (send *pr2* arm :inverse-kinematics pose-vector :debug-view :t)
  )

(defun boundingbox-array-callback(msg)
  (when (car (send msg :boxes))
    (setq target-box (car (send msg :boxes)))
    (setq dimension (send target-box :dimensions))
    (setq pose (send target-box :pose))

    (setq position (send pose :position))
    
    (setq target-position (make-coords))
    (send target-position :replace-pos (float-vector (send position :x) (send position :y) (send position :z)))

    (unix:usleep (* 1000 5000))

    (ros::ros-warn "time-out")
    
    (check-arm-reachability target-position :larm)
    (send-target-pose-to-robot target-position :larm 15000)
     
    
    (setq x (send (send target-box :dimensions) :x))

    (print target-position)
    (ros::ros-warn "subscribing")
    )
  )


(defun main()
  (manip-pose)
  (ros::subscribe "/cluster_decomposer_final/boxes"
		  jsk_recognition_msgs::BoundingBoxArray #'boundingbox-array-callback)
  (do-until-key
  (ros::spin-once))
  )

;; (main)


;; test using au-kun code

(defun push-test ()
  (load "move-objects.l")
  (init-scene)
  (while (ros::ok)
    (ros::spin-once)
    (ros::ros-warn "Robot received an order, need to work now!")
    (unix:usleep (* 1000 5000))
    (update-scene)
    (push-one-time)
    (ros::sleep))
  )
