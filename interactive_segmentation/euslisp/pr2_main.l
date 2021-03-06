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
(load "move-objects.l")
;; (load "grasp-object.l")

(defvar *thresh-x* 10.0)
(defvar *thresh-y* 10.0)
(defvar *thresh-z* 10.0)

(ros::ros-info "RUNNING")

(defun manip-pose(&optional(time 12000))
  (send *pr2* :angle-vector *manip-pose*)
  (if (boundp '*ri*)
      (send *ri* :angle-vector (send *pr2* :angle-vector) time)
    (send *ri* :wait-interpolation)
    )
  )


(defun main()
  (manip-pose)
  (ros::subscribe *boundingbox-array-topic*
		  jsk_recognition_msgs::BoundingBoxArray #'boundingbox-array-callback)
  (do-until-key
  (ros::spin-once))
  )


(defun push-test ()
  ;; (load "move-objects.l")
  (init-scene)
  (if (update-scene)
      (progn
	(ros::ros-warn "Robot received an order, need to work now!")
	(push-one-time)
	)
    (ros::ros-warn "CAN NOT PUSH CURRENT BOX")
    )
  (ros::ros-warn "Robot completed execution!")
  (report-to-recognition-node)
  )

					;(while (ros::ok)
(defun push-node()
  (do-until-key
   (ros::spin-once)
   (push-test)
   (ros::sleep)
   )
  )
