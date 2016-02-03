#!/usr/bin/env roseus

(load "pr2_main.l")


(setq *goal-pos* (make-coords :rpy (float-vector 0 (deg2rad 90) 0))))
(send *goal-pos* :replace-pos #f(600 0 700))

(defun generate-scene-model()
  (setq table (make-cube 500 1000 700))
  (send table :translate #f(700 0 350))
  (send table :set-color #f(0.87 0.72 0.53))
  (objects table)
  )

(defun collision-check(box &optional (arm :larm))
  (let (res)
    (dolist (link (append (send *pr2* arm :links)
			  (send (send (send *pr2* arm :end-coords :parent) :parent) :child-links)))
      ;; (dolist (box *boxes*)
      (setq res (pqp-collision-check link box))
      (ros::ros-info (format nil "Checking :~A~%" link))
        (when (eq res 1)
          (ros::ros-fatal (format nil "check arm collision failed:~A~%" res))
          (return-from collision-check nil))
	;; )
	) t)
  (ros::ros-warn "-- no collision detected")
  )


(defun send-target-pose-to-robot(pose-vector &optional (arm :larm) &optional (time 5000))
  (ros::ros-warn "-- moving the robot")
  (when (boundp '*ri*)
    (send *ri* :angle-vector (send *pr2* :angle-vector) time)
    (send *ri* :wait-interpolation)
    )
  )

(defun check-arm-reachability(pose-vector &optional (arm :larm))
  (when (send *pr2* arm :inverse-kinematics pose-vector :rotation-axis :x :debug-view :t)
    (ros::ros-warn "-- ik solved... checking for collision")
    (if (eq (collision-check table) t)
    	(send-target-pose-to-robot pose-vector arm 5000)
      (ros::ros-error "-- collision detected")
      )
    )
  )



(defun move-to-manip-position(pose-vector &optional (arm :larm))
  (ros::ros-info "moving arm to manipulation position")
  )
