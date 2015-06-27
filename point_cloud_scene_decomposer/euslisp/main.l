(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "point_cloud_scene_decomposer")
(ros::roseus "robot_pushing_motion_node")
(ros::rate 5)

(setq *polygon-array-topic* "/multi_plane_estimate/output_refined_polygon")
;;(setq *boundingbox-array-topic* "/cluster_decomposer_final/boxes")
(setq *boundingbox-array-topic* "/point_cloud_scene_decomposer_cluster_points/boxes")


(require "package://pr2eus/pr2-interface.l")
(load "generate-tabletop-scene.l")
(load "move-objects.l")

(if (not (boundp '*pr2*))
    (pr2-init))
(if (not (boundp '*irtviewer*))
    (make-irtviewer))
(send *irtviewer* :objects *pr2*)

(setq *signal-stamp* nil)
(setq *command* nil)
(setq *counter* nil)
(setq *private-counter* 0)

(defun signal-cb (msg)
  (setq *signal-stamp* (send msg :header :stamp))
  (setq *command* (send msg :command))
  (setq *counter* (send msg :counter)))

(defun reset-signal()
  (setq *signal-stamp* nil)
  (setq *command* nil)
  (setq *counter* nil)
  (setq *private-counter* 0))

(defun main ()
  (init-scene)
  (while (ros::ok)
    (ros::spin-once)
    (when (and (eq *command* 2) (eq *counter* *private-counter*))
      (ros::ros-warn "Robot received an order, need to work now!")
      (unix:usleep (* 1000 5000))
      (update-scene)
      (push-one-time)
      (let ((report-signal (instance point_cloud_scene_decomposer::signal :init)))
        (send report-signal :header :stamp (ros::time-now))
        (send report-signal :command 3)
        (send report-signal :counter *counter*)
        (incf *private-counter*)
        (unix:usleep (* 1000 10000))
        (ros::ros-warn "Robot Finish working, send a report!")
        (report-to-recognition-node)
        (ros::publish "/scene_decomposer/output/signal" report-signal)))
    (ros::sleep)))

(ros::advertise "/scene_decomposer/output/signal" point_cloud_scene_decomposer::signal 1)
(ros::subscribe "/scene_decomposer/output/signal" point_cloud_scene_decomposer::signal #'signal-cb)

        
      

      



      
        
    
