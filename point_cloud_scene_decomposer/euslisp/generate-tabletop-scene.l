(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "point_cloud_scene_decomposer")
(ros::roseus "robot_pushing_motion_node")

(require "package://pr2eus/pr2-interface.l")
(if (not (boundp '*pr2*))
    (pr2-init))
(if (not (boundp '*irtviewer*))
    (make-irtviewer))
(send *irtviewer* :objects *pr2*)

(setq *table* nil)
(setq *boxes* nil)
(setq *left-arm-candidates* nil)
(setq *right-arm-candidates* nil)
(setq *push-style* :use-fingertip) ;;push-stype :use-fingertip / use-palm
(setq *pushed-box* nil)
(setq *manip-pose* #f(131.783 54.9158 -1.3071 0.272882 -88.7191 -357.653 -5.85497 360.042 -45.3354 1.53671 2.39065 -95.778 -1.10987 -5.93392 359.975 1.67125 48.912))

;;#f(117.164 54.9158 -1.3071 0.272882 -88.7191 -357.647 -5.78018 360.042 -45.3354 1.53671 2.39065 -95.778 -1.10987 -5.93392 359.975 1.70125 46.872) lower pose
;;#f(296.291 54.9158 -1.3071 0.272882 -88.7191 -357.647 -5.7403 360.042 -45.3354 1.53671 2.39065 -95.778 -1.10987 -5.91148 359.972 7.46125 44.208) higher pose

(send *ri* :angle-vector *manip-pose*)
(send *ri* :start-grasp)
(setq *tfl* (instance ros::transform-listener :init))

;;report to recognition node
(ros::advertise "/robot_pushing_motion_node/pushed_box_index" std_msgs::Int64 1)
(ros::advertise "/robot_pushing_motion_node/pushed_box_assumed_pose" geometry_msgs::PoseStamped 1)


(defun report-to-recognition-node ()
  (let ((idx (instance std_msgs::Int64 :init))
        (final-pose (instance geometry_msgs::PoseStamped :init))
        trans)
    (if *pushed-box*
        (progn
          (send idx :data (position *pushed-box* (send *gba* :get-all-boxes)))
          (send final-pose :header :stamp (ros::time-now))
          (send final-pose :header :frame_id "/openni_rgb_optical_frame")
          (setq trans (send *tfl* :lookup-transform  "/openni_rgb_optical_frame" "/base_footprint"(ros::time 0)))
          (send final-pose :pose (ros::coords->tf-pose (send (send *pushed-box* :copy-worldcoords) :transform trans :world)))
          (ros::publish "/robot_pushing_motion_node/pushed_box_index" idx)
          (ros::publish "/robot_pushing_motion_node/pushed_box_assumed_pose" final-pose))
      (progn
        (send idx :data -1)
        (send final-pose :header :stamp (ros::time-now))
        (send final-pose :header :frame_id "/openni_rgb_optical_frame")
        (ros::publish "/robot_pushing_motion_node/pushed_box_index" idx)
        (ros::publish "/robot_pushing_motion_node/pushed_box_assumed_pose" final-pose)))))

  


(defclass generate-polygon-array
  :slots 
  (*vertices-list-org* *plane-normal-list* *latest-polygon-list* *polygon-list* *cloud-stamp* *cloud-centroid-list* *plgs-cb-flag* *cofs-cb-flag*))

(defmethod generate-polygon-array
  (:init 
   ()
   (ros::subscribe *polygon-array-topic* jsk_recognition_msgs::PolygonArray #'send self :polygon-array-cb))

  (:polygon-array-cb
   (msg)
   (ros::ros-info "polygons callback")
   (setq *plgs-cb-flag* t)
   (setq *cloud-stamp* (send msg :header :stamp))
   (let ((plg-list (send msg :polygons)))

     (setq *vertices-list-org* (mapcar #'(lambda (plg)
                                           (let ((pts (send (send plg :polygon) :points))
                                                 pts-list
                                                 trans
                                                 trans-pt)
                                             (setq trans (send *tfl* :lookup-transform "/base_footprint" "/openni_rgb_optical_frame" (ros::time 0)))
                                             (dolist (pt pts)
                                               (setq trans-pt (send trans :transform-vector
                                                                    (scale 1000 (float-vector (send pt :x) (send pt :y) (send pt :z)))))
                                               (push trans-pt pts-list))
                                             (reverse pts-list)))
                                       plg-list))
     ;; (dotimes (i (length *vertices-list-org*))
     ;;   (let* ((c (send *tfl* :lookup-transform "/base_footprint" (format nil "/multi_plane_centroidoutput~0,2D" i) *cloud-stamp*)))
     ;;     (when c
     ;;       (send c :transform (send *pr2* :copy-worldcoords) :world)
     ;;       (push  (send c :copy-worldcoords) centroid-list))))
     ;; (setq *cloud-centroid-list* (reverse centroid-list))
     (send self :init-eus-polygons)))
  
  (:init-eus-polygons
   ()
   ;; (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
   ;; (send *irtviewer* :objects (list *pr2*))
   (let (polygons-list)
     ;;shift the detected polygons a little considering the thickness
     (dotimes (i (length *vertices-list-org*))
	(let ((plg-temp (instance polygon :init :vertices (elt *vertices-list-org* i)))
	      plg v)

	  ;; (if (>= (v. (matrix-column (send *pr2* :head :end-coords :worldrot) 2) (send plg-temp :normal)) 0) ;;shift for table thickness
	  ;;     (setq v (send plg-temp :normal))
	  ;;   (setq v (scale -1.0 (send plg-temp :normal))))
	  (setq plg (instance polygon :init :vertices (send plg-temp :vertices)))
	  ;; (setq plg (instance polygon :init :vertices (mapcar #'(lambda (pt) (v+ pt (scale 25 v))) (send plg-temp :vertices))))
	  (push plg polygons-list)))

      (setq *latest-polygon-list* (reverse polygons-list))))

  (:draw-polygons
   (plgs)
   (mapcar #'(lambda (plg ct)
               (send plg :reset-normal)
               (let* (;;(color (float-vector (random 1.0) (random 1.0) (random 1.0)))
                      (color #f(0.87 0.72 0.53)) ;;burlywood color
                      (prism (make-prism (mapcar #'(lambda (p) (scale 1 p)) (send plg :vertices)) (send plg :normal)))
                      (sphere (make-sphere 20)))
                 (send sphere :move-to ct :world)
                 (send sphere :set-color :white)
                 (send prism :set-color color)
                 ;; (send *irtviewer* :objects sphere)
                 (send *irtviewer* :objects prism)))
           plgs
           ;; (if (equal (length *cloud-centroid-list*) (length polygons-list))
           ;;     (progn (print "check1")
           ;;            *cloud-centroid-list*)
           ;;   (progn (print "check2")
           (mapcar #'(lambda (p) (make-coords :pos (elt (send p :centroid) 1))) plgs))
   (send *irtviewer* :draw-objects :flush t)
   )
  
  (:sort-polygons-by-dist-from-head
   (&optional (plgs *polygon-list*) (pt (send (car (send *pr2* :head)) :worldpos)))
   (sort plgs #'(lambda (plg1 plg2)
                  (if (<= (norm (v- (elt (send plg1 :centroid) 1) pt)) (norm (v- (elt (send plg2 :centroid) 1) pt)))
                      t))))
   

  (:update-polygons
   ()
   (setq *polygon-list* (copy-object *latest-polygon-list*)))

  (:get-polygons
   ()
   *polygon-list*)

  )

(defclass generate-boundingbox-array
  :slots
  (*latest-box-list* *box-list* *boxs-stamp*))

(defmethod generate-boundingbox-array
  (:init
   ()
   (ros::subscribe *boundingbox-array-topic* jsk_recognition_msgs::BoundingBoxArray #'send self :boundingbox-array-cb))

  (:boundingbox-array-cb
   (msg)
   (ros::ros-info "boundingbox callback")
   (setq *boxes-stamp* (send msg :header :stamp))

   (let ((boxes (send msg :boxes))
         (trans (send *tfl* :lookup-transform "/base_footprint" "/openni_rgb_optical_frame" (ros::time 0))))
     (setq *latest-box-list*
           (mapcar #'(lambda (box)
                       (let* ((dim (ros::tf-point->pos (send box :dimensions)))
                              (cube (make-cube (elt dim 0) (elt dim 1) (elt dim 2)))
                              (cds (ros::tf-pose->coords (send box :pose))))
                         (send cds :transform trans :world)
                         (send cube :move-to cds)
                         (send cube :set-color (float-vector (random 1.0) (random 1.0) (random 1.0)))
                         cube))
                   boxes))))

  (:draw-boxes
   (boxes)
   (send *irtviewer* :objects (append (send *irtviewer* :objects) boxes))
   (send *irtviewer* :draw-objects :flush t))

  (:update-boxes
   ()
   (setq *box-list* (copy-object *latest-box-list*)))

  (:get-all-boxes
   ()
   *box-list*)
  
  (:get-tabletop-boxes
   ()
   *filter-boxes*)

  (:filter-tabletop-boxes
   ()
   (let (boxes-temp
         ground-pos)
     (dolist (b *box-list*)
       ;;filter boxes that are not on table
       (setf (elt (setq ground-pos (send (send b :copy-worldcoords) :worldpos)) 2) 0)
       (if (send *table* :intersect-line (send b :worldpos) ground-pos)
         ;; (if (send *table* :intersect-point-vector (send b :worldpos) #f(0 0 -1))
         ;;     (if (> (elt (send b :worldpos) 2) (elt (cadr (send *table* :centroid)) 2))
           (push b boxes-temp)))
     (setq *filter-boxes* (reverse boxes-temp))))

  (:boxes-centroid
   ()
   (let ((sum #f(0 0 0)))
     (dolist (b *boxes*)
       (setq sum (v+ sum (send b :worldpos))))
     (setq sum (scale (/ 1.0 (length *boxes*)) sum))
     sum))

  )


(defun init-scene ()
  (setq *gpa* (instance generate-polygon-array :init))
  (setq *gba* (instance generate-boundingbox-array :init)))

(defun update-scene ()
  (setq *pushed-box* nil)
  (send *pr2* :angle-vector (send *ri* :state :potentio-vector))
  (send *irtviewer* :objects (list *pr2*))
  (setq *table* nil)
  (setq *boxes* nil)
  (while t
    (ros::spin-once)
    (if (and (send *gpa* :update-polygons) (send *gba* :update-boxes))
        (return)
      (ros::spin-once)))
  (send *gpa* :sort-polygons-by-dist-from-head)
  (setq *table* (car (send *gpa* :get-polygons)))
  (send *gba* :filter-tabletop-boxes)
  (setq *boxes* (send *gba* :get-tabletop-boxes))

  (send *gpa* :draw-polygons (list *table*))
  (send *gba* :draw-boxes *boxes*)
  (setq *left-arm-candidates* (sort-boxes-by-dist-from *boxes* (send *pr2* :larm :end-coords)))
  (setq *right-arm-candidates* (sort-boxes-by-dist-from *boxes* (send *pr2* :rarm :end-coords)))
  (setq *boxes-small-to-big* (sort (copy-seq *boxes*) #'(lambda (b1 b2)
                                                        (if (<= (send b1 :volume) (send b2 :volume))
                                                            t))))
  (setq *boxes-on-table* nil)
  (dolist (b *boxes*)
    (if (<= (smallest-dist-to-table b) 50)
        (push b *boxes-on-table*))))


(defun smallest-dist-to-table (box &optional (table *table*))
  (let ((smallest-dist 10000))
    (dolist (f (send box :faces))
      (let (dist)
        (setq dist (abs (- (elt (cadr (send f :centroid)) 2) (elt (cadr (send *table* :centroid)) 2))))
        (if (<= dist smallest-dist)
            (setq smallest-dist dist))))
    smallest-dist))

(defun sort-boxes-by-dist-from (boxes cds)
  (let ((sorted-boxes (copy-seq boxes)))
    (sort sorted-boxes #'(lambda (b1 b2)
                           (if (<= (abs (norm (v- (send cds :worldpos) (send b1 :worldpos))))
                                   (abs (norm (v- (send cds :worldpos) (send b2 :worldpos)))))
                               t)))
    sorted-boxes))




