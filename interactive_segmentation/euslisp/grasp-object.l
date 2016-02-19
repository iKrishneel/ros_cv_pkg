#!/usr/bin/env roseus

;(require "package://pr2eus/pr2-interface.l")
(require "package://pr2eus_moveit/euslisp/pr2eus-moveit.l")

(ros::load-ros-manifest "interactive_segmentation")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "geometry_msgs")

; (load "pr2_main.l")

(setq *parent-link* "/base_footprint")
(setq *child-link* "/head_mount_kinect_rgb_optical_frame")
(setq *tf-listener* (instance ros::transform-listener :init))

(defclass handle-object
  :slots
  (*grasp-obj-tgt-cds*
   *grasp-obj-tgt-pre-cds*
   *last-update-time*
   *trans*
   )
  )

(defmethod handle-object
  (:init ()
	 ;; (setq *trans* nil)
	 ;; (setq *grasp-obj-tgt-pre-cds* nil)
	 ;; (setq *last-update-time* (ros::time))
	 ;; (setq *grasp-obj-tgt-cds* nil)
	 (setq *trans* (send *tf-listener* :lookup-transform *parent-link* *child-link* (ros::time 0)))
	 (ros::subscribe "/handle_estimator/output_best" geometry_msgs::PoseStamped
			 #'send self :tgt-pose-cb)
	 (ros::subscribe "/handle_estimator/output_best_preapproach" geometry_msgs::PoseStamped
			 #'send self :tgt-pre-poses-cb)
	 )

  (:tgt-pose-cb (msg)
		(unless *trans*
		  (return-from :tgt-pose-cb))
		(setq *last-update-time* (send msg :header :stamp))
		(setq *grasp-obj-tgt-cds* (send (ros::tf-pose->coords (send msg :pose)) :translate #f(0 0 0)))
		(send *grasp-obj-tgt-cds* :transform *trans* :world)

		(send *grasp-obj-tgt-cds* :translate #f(15 0 0) :world)
		
		(send *irtviewer* :objects  *grasp-obj-tgt-cds*)   
		(send *irtviewer* :draw-objects :flush t)
		)
  
  (:tgt-pre-poses-cb (msg)
		     (unless *trans*
		       (return-from :tgt-pre-poses-cb))
		     (print *trans*)
		     (setq *grasp-obj-tgt-pre-cds* (ros::tf-pose->coords (send msg :pose)))
		     (send *grasp-obj-tgt-pre-cds* :transform *trans* :world)		     
		     (send *irtviewer* :objects  *grasp-obj-tgt-pre-cds*)
		     (send *irtviewer* :draw-objects :flush t)
		     )
  
  (:grasp-selected-object (&optional (use-arm :larm))
			  (unless *grasp-obj-tgt-cds*
			    (ros::ros-error "empty list")
			    (return-from :grasp-selected-object))
			  (unless *grasp-obj-tgt-pre-cds*
			    (ros::ros-error "empty list pre")
			    (return-from :grasp-selected-object))
			  			  
			  (let ((avs (send self :calc-avs-to-grasp-selected-object use-arm)))
			    (when (not (= (length avs) 2))
			      (return-from :grasp-selected-object nil))
			    (send *ri* :angle-vector (car avs) 4000)
			    (send *ri* :stop-grasp use-arm)
			    (send *ri* :wait-interpolation)
			    (send *ri* :angle-vector (cadr avs) 4000)
			    (send *ri* :wait-interpolation)
			    (send *ri* :start-grasp use-arm :gain 0.01)
			    (send *ri* :wait-interpolation)
			    )

			  (setq *grasp-obj-tgt-cds* nil)
			  (setq *grasp-obj-tgt-pre-cds* nil)
			  
			  t)
  
 (:release-object (&optional (use-arm :larm))
		  (send *ri* :stop-grasp use-arm)
		  (unix:usleep (* 1000 1000))
		  (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
		  (unless (send *pr2* use-arm :inverse-kinematics (send (send *pr2* use-arm
									      :end-coords
									      :copy-worldcoords)
									:translate #f(-100 0 0)))
		    (send *pr2* use-arm :inverse-kinematics (send (send *pr2* use-arm
									:end-coords
									:copy-worldcoords)
								  :translate #f(-50 0 0))))
		  (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
		  (send *ri* :wait-interpolation)
		  )

 (:calc-avs-to-grasp-selected-object (&optional (use-arm :larm))
				     (let ((tgt-cds (send *grasp-obj-tgt-cds* :copy-worldcoords))
					   (pre-cds (send *grasp-obj-tgt-pre-cds* :copy-worldcoords))
					   avs)
				       (if (send *pr2* :inverse-kinematics
						 pre-cds
						 :move-target (send *pr2* use-arm :end-coords)
						 :link-list (send *pr2* :link-list
								  (send *pr2* use-arm :end-coords :parent))
						 :use-torso t
						 :rotational-axis t)
					   (progn
					     (push (send *pr2* :angle-vector) avs)
					     (if (send *pr2* :inverse-kinematics
						       tgt-cds
						       :move-target (send *pr2* use-arm :end-coords)
						       :link-list (send *pr2* :link-list
									(send *pr2* use-arm :end-coords :parent))
						       :use-torso t
						       :rotational-axis t)
						 (push (send *pr2* :angle-vector) avs)))
					 (progn
					   (if (send *pr2* :inverse-kinematics
						     (send pre-cds :rotate (deg2rad 180) :x)
						     :move-target (send *pr2* use-arm :end-coords)
						     :link-list (send *pr2* :link-list
								      (send *pr2* use-arm :end-coords :parent))
						     :use-torso t
						     :rotational-axis t)
					       (push (send *pr2* :angle-vector) avs))
					   (if (send *pr2* :inverse-kinematics
						     (send tgt-cds :rotate (deg2rad 180) :x)
						     :move-target (send *pr2* use-arm :end-coords)
						     :link-list (send *pr2* :link-list
								      (send *pr2* use-arm :end-coords :parent))
						     :use-torso t
						     :rotational-axis t)	
					       (push (send *pr2* :angle-vector) avs))))
				       (if (= (length avs) 2)
					   (reverse avs)
					 (progn
					   (ros::ros-warn "failed to solve ik to grasp object")
					   nil))
				       )
				     )
 )


(defun init-grasp ()
  (setq *handle-obj* (instance handle-object :init))
  )

(defun lift-up(&optional (use-arm :larm))

  ;; (send *ri* :start-grasp use-arm :gain 0.01)
					;(speak-en "object grasped")
  (send *ri* :wait-interpolation)
  
  (setq trans-up-pos #f(0 0 100))
  (setq endpos (send *pr2* :larm :end-coords :copy-worldcoords))
  (send endpos :translate trans-up-pos :world)
  (send *pr2* use-arm :inverse-kinematics  endpos)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)

  (move-object)

  ;(speak-en "releasing object")
  (setq trans-down-pos #f(0 0 -100))
  (setq endpos (send *pr2* :larm :end-coords :copy-worldcoords))
  (send endpos :translate trans-down-pos :world)
  (send *pr2* use-arm :inverse-kinematics  endpos)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :wait-interpolation)

  (send *ri* :stop-grasp use-arm)
  (send *ri* :wait-interpolation)

  (unix:usleep (* 1000 2000))

  ;; move-hand up
  (setq trans-up-pos #f(0 0 200))
  (setq endpos (send *pr2* :larm :end-coords :copy-worldcoords))
  (send endpos :translate trans-up-pos :world)
  (send *pr2* use-arm :inverse-kinematics  endpos)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)

  (send *ri* :angle-vector (send *pr2* :angle-vector *manip-pose*) 2000)
  (send *ri* :wait-interpolation)
  )

(defun move-object(&optional (use-arm :larm) )
  ;(speak-en "Moving Object to new position")
  (setq endpos (send *pr2* :larm :end-coords :copy-worldcoords))
  (send endpos :translate #f(0 400 0) :world)
  (send *pr2* use-arm :inverse-kinematics  endpos)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  )

(defun grasp-object()
  (init-grasp)
  ;; (while (ros::ok)
  (ros::spin-once)
  (setq *use-arm* :larm)
  (setq process (send *handle-obj* :grasp-selected-object *use-arm*))
  (when process
    (lift-up *use-arm*))
  (unix:usleep (* 1000 3000))

    ;; (manip-pose)
  ;; )
  t)
