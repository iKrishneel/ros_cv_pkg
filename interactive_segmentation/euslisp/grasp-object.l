#!/usr/bin/env roseus

(require "package://pr2eus/pr2-interface.l")
(require "package://pr2eus_moveit/euslisp/pr2eus-moveit.l")
;; (pr2-init)

(ros::load-ros-manifest "interactive_segmentation")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "geometry_msgs")

;; (unless (boundp '*irtviewer*)
;;   (make-irtviewer))
;; (send *irtviewer* :objects *pr2*)


(defclass handle-object
  :slots
  (*grasp-obj-tgt-cds*
   *grasp-obj-tgt-pre-cds-array*
   *grasp-obj-tgt-pre-cds*
   *last-update-time*
   )
  )

(defmethod handle-object
  (:init
   ()
   (setq *grasp-object-tgt-cds* nil)
   (setq *grasp-obj-tgt-pre-cds-array* nil)
   (setq *grasp-obj-tgt-pre-cds* nil)
   (setq *last-update-time* (ros::time))
   (ros::subscribe "/handle_estimator/output_best"
		   geometry_msgs::PoseStamped
		   #'send self :tgt-pose-cb)
   ;; (ros::subscribe "/handle_estimator/output_best_preapproach"
   ;; 		   geometry_msgs::PoseStamped
   ;; 		   #'send self :tgt-pre-poses-cb)
   )
   


  (:tgt-pose-cb 
   (msg)
   (setq *last-update-time* (send msg :header :stamp))
   (setq *grasp-obj-tgt-cds* (send (ros::tf-pose->coords (send msg :pose)) :translate #f(10 0 0)))
   (send *irtviewer* :objects  *grasp-obj-tgt-cds*)

   (print "target-pose-cb")
   (print *grasp-obj-tgt-cds*)
   
   (send *irtviewer* :draw-objects :flush t)
   )
  
  (:tgt-pre-poses-cb 
   (msg)
   (setq *grasp-obj-tgt-pre-cds* (ros::tf-pose->coords (send msg :pose)))
   (send *irtviewer* :objects  *grasp-obj-tgt-pre-cds*)
   (send *irtviewer* :draw-objects :flush t)
   )
  
  (:grasp-selected-object 
   (&optional (use-arm :larm))
   ;; (unless (ros::time= *last-update-time* *request-time*)
   ;;   (ros::ros-warn "latest received pose is to old, give up")
   ;;   (return-from :grasp-selected-object nil))

     (let ((avs (send self :calc-avs-to-grasp-selected-object use-arm)))
       (when (not (= (length avs) 2))
	 (return-from :grasp-selected-object nil))
       (send *ri* :angle-vector (car avs) 4000)
       (send *ri* :stop-grasp use-arm)
       (send *ri* :wait-interpolation)
       (send *ri* :angle-vector (cadr avs) 1000)
       (send *ri* :wait-interpolation)
       (send *ri* :start-grasp use-arm :gain 0.1)
       (send *ri* :wait-interpolation))
     t)
  
 (:release-object
   (&optional (use-arm :larm))
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

(:calc-avs-to-grasp-selected-object 
 (&optional (use-arm :larm))
 (let ((tgt-cds (send *grasp-obj-tgt-cds* :copy-worldcoords))
       (pre-cds (send *grasp-obj-tgt-pre-cds* :copy-worldcoords))
       avs)
   (if (send *pr2* :inverse-kinematics
	     pre-cds
	     :move-target (send *pr2* use-arm :end-coords)
	     :link-list (send *pr2* :link-list (send *pr2* use-arm :end-coords :parent))
	     :use-torso t
	     :rotational-axis t)
       (progn
	 (push (send *pr2* :angle-vector) avs)
	 (if (send *pr2* :inverse-kinematics
	       tgt-cds
	       :move-target (send *pr2* use-arm :end-coords)
	       :link-list (send *pr2* :link-list (send *pr2* use-arm :end-coords :parent))
	       :use-torso t
	       :rotational-axis t)
	 (push (send *pr2* :angle-vector) avs)))
     (progn
       (if (send *pr2* :inverse-kinematics
	     (send pre-cds :rotate (deg2rad 180) :x)
	     :move-target (send *pr2* use-arm :end-coords)
	     :link-list (send *pr2* :link-list (send *pr2* use-arm :end-coords :parent))
	     :use-torso t
	     :rotational-axis t)
	   (push (send *pr2* :angle-vector) avs))
       (if (send *pr2* :inverse-kinematics
	     (send tgt-cds :rotate (deg2rad 180) :x)
	     :move-target (send *pr2* use-arm :end-coords)
	     :link-list (send *pr2* :link-list (send *pr2* use-arm :end-coords :parent))
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
