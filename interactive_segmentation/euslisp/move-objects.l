(defun calc-push-coords-list (box)
  (let (cds-list
        cds-p cds-n)
    ;;x direction
    (setq cds-p (send box :copy-worldcoords))
    (setq cds-n (send box :copy-worldcoords))
    (send cds-p :translate (float-vector (/ (elt (car (send box :csg)) 1) 2) 0 0))
    (send cds-n :translate (float-vector (/ (* -1 (elt (car (send box :csg)) 1)) 2) 0 0))
    (send cds-p :rotate (deg2rad 180) :z)
    (push cds-p cds-list)
    (push cds-n cds-list)
    ;; y direction
    (setq cds-p (send box :copy-worldcoords))
    (setq cds-n (send box :copy-worldcoords))
    (send cds-p :translate (float-vector 0 (/ (elt (car (send box :csg)) 2) 2) 0))
    (send cds-n :translate (float-vector 0 (/ (* -1 (elt (car (send box :csg)) 2)) 2) 0))
    (if (>= (abs (rad2deg (vector-angle (send box :z-axis) #f(0 0 1)))) 90)
        (progn
          (send cds-p :rotate (deg2rad -90) :z)
          (send cds-n :rotate (deg2rad 90) :z))
      (progn
        (send cds-p :rotate (deg2rad 90) :z)
        (send cds-n :rotate (deg2rad -90) :z)))
    (push cds-p cds-list)
    (push cds-n cds-list)
    cds-list))

(defun check-collision (&optional (arm :larm))
  (let (res)
    (dolist (link (append (send *pr2* arm :links) (send (send (send *pr2* arm :end-coords :parent) :parent) :child-links)))
      (dolist (box *boxes*)
        (setq res (pqp-collision-check link box))
        (when (eq res 1)
          (ros::ros-warn (format nil "check arm collision failed:~A~%" res))
          (return-from check-collision nil))))
  t))


(defun filter-pushing-cds (target-box other-boxes &optional (arm :larm))
  (let ((insert-offset 30)
        target-cds-list
        filter-target-cds-list)

    
    (ros::ros-warn (format nil "checking box:~A~%" target-box))
    ;;filter out if box not on table
    (unless (member target-box *boxes-on-table*)
      (return-from filter-pushing-cds))

    (setq target-cds-list (mapcar #'(lambda (cds)
                                      (send cds :translate (float-vector (* -1 insert-offset) 0 0))
                                      (cond ((eq *push-style* :use-palm)
                                             (send cds :rotate (deg2rad 90) :y)
                                             (send cds :rotate (deg2rad 180) :z))
                                            ((eq *push-style* :use-fingertip)
                                             (send cds :rotate (deg2rad 90) :x))))
                                      (calc-push-coords-list target-box)))

    ;;filter target-cds-list
    (dolist (cds target-cds-list)
      (print cds)
        (let ((cds-rotate? (list nil 0))
              ik-solved
              collided)

          ;;not moving toward centrod
          (if (>= (abs (rad2deg (vector-angle (send cds :z-axis) (v- (send *gba* :boxes-centroid) (send cds :worldpos))))) 80)
              ;;arm will not collide
              (progn
                (print "toward centroid checked")
                (setq ik-solved (send *pr2* :inverse-kinematics cds :move-target (send *pr2* arm :end-coords) :link-list (send *pr2* :link-list (send *pr2* arm :end-coords :parent)) :use-torso 0 :rotation-axis t :debug-view nil :warnp nil))
                (if (not ik-solved)
                    (progn
                      (print "ik-not-solved")
                      (setq ik-solved (send *pr2* :inverse-kinematics (send (send cds :copy-worldcoords) :rotate (deg2rad 180) :x) :move-target (send *pr2* arm :end-coords) :link-list (send *pr2* :link-list (send *pr2* arm :end-coords :parent)) :use-torso 0 :rotation-axis t :debug-view nil :warnp nil))))
                (send *irtviewer* :draw-objects :flush t)
                (if (and ik-solved (check-collision arm))
                    (progn
                      (setf (elt cds-rotate? 1) 1)
                      (print "arm collision checked")
                      ;;no obstacle on path
                      (let ((tmp-box (copy-object target-box)))
                        (send tmp-box :set-color :yellow)
                        (send target-box :set-color :red)
                        (send tmp-box :translate (if (eq *push-style* :use-palm) #f(0 0 50) #f(50 0 0)) cds)
                        (print "hello")
                        (objects tmp-box)
                        (if nil ;;other-boxes ;;if there are other boxes
                            (progn
                              (print "there are other boxes")
                              (dolist (ob other-boxes)
                                (setq collided (if (> (pqp-collision-check tmp-box ob) 0) t nil))
                                (if collided (return)))))
                        (if (not collided) ;;no collision
                            (progn
                              (send tmp-box :translate (if (eq *push-style* :use-palm) #f(0 0 50) #f(50 0 0)) cds)
                              (send *irtviewer* :draw-objects :flush t)
                              (setq collided nil)
                              (if other-boxes
                                  (dolist (ob other-boxes)
                                    (setq collided (if (and (> (pqp-collision-check tmp-box ob) 0) t nil)))
                                    (if collided (return))))
                              (if t ;;(not collided) ;;no collison
                                  (progn
                                    (print "on path collision checked")
                                    (setf (elt cds-rotate? 0) cds)
                                    (push cds-rotate? filter-target-cds-list)))))
                        (send *irtviewer* :objects (delete tmp-box (send *irtviewer* :objects) :count 1)))))))))
    filter-target-cds-list))

(defun push-until-stop ()
  (dotimes (i 10)
    (if (= (mod i 2) 0)
        (setq *push-style* :use-fingertip)
      (setq *push-style* :use-palm))
    (let ((sp (make-sphere 20 :pos (send *gba* :boxes-centroid))))
      (send *irtviewer* :objects sp)
      (dolist (target-box *left-arm-candidates*)
        (let ((other-boxes (remove target-box *left-arm-candidates*)))
          (push-boxes target-box other-boxes :larm)
          (send sp :move-to (make-coords :pos (send *gba* :boxes-centroid)) :world)))
    
      (send *pr2* :angle-vector *manip-pose*)
      (send *ri* :angle-vector *manip-pose* 5000)
      (send *ri* :wait-interpolation)
      
      (unix:usleep (* 1000 8000))
      (update-scene)
      (dolist (target-box *right-arm-candidates*)
	(let ((other-boxes (remove target-box *right-arm-candidates*)))
	  (push-boxes target-box other-boxes :rarm)
	  (send sp :move-to (make-coords :pos (send *gba* :boxes-centroid)) :world)))
      
      (send *pr2* :angle-vector *manip-pose*)
      (send *ri* :angle-vector *manip-pose* 5000)
      (send *ri* :wait-interpolation)
      (unix:usleep (* 1000 8000))
      (update-scene))))

(defun push-one-time ()
  (setq *pushed-box* nil)
  (setq *push-style* :use-fingertip)

  (let ((sp (make-sphere 20 :pos (send *gba* :boxes-centroid))))
    (send *irtviewer* :objects sp)
    (dolist (target-box *boxes-small-to-big*)
      (let ((other-boxes (remove target-box *boxes-small-to-big*)))
        (push-boxes target-box other-boxes :larm)
        (send sp :move-to (make-coords :pos (send *gba* :boxes-centroid)) :world)
        (if *pushed-box* (return))))
    
      (send *pr2* :angle-vector *manip-pose*)
      (send *ri* :angle-vector *manip-pose* 5000)
      (send *ri* :wait-interpolation)
      
      (unless *pushed-box*
        (dolist (target-box *boxes-small-to-big*)
          (let ((other-boxes (remove target-box *boxes-small-to-big*)))
            (push-boxes target-box other-boxes :rarm)
            (send sp :move-to (make-coords :pos (send *gba* :boxes-centroid)) :world)
            (if *pushed-box* (return))))
        
        (send *pr2* :angle-vector *manip-pose*)
        (send *ri* :angle-vector *manip-pose* 5000)
        (send *ri* :wait-interpolation))))

(defun push-boxes (target-box other-boxes &optional (arm :larm))
  (let (final-cds-rotate?-list
        av-list)
    ;;     target-box
    ;;     other-boxes)
    ;; (setq target-box (car *left-arm-candidates*))
    ;; (setq other-boxes (cdr *left-arm-candidates*))
    (setq final-cds-rotate?-list (filter-pushing-cds target-box other-boxes arm))
    
    (when final-cds-rotate?-list
      (print "robot-start-moving")
      (print final-cds-rotate?-list)
      (let* ((obj-cds (make-cascoords :coords (send (car (car final-cds-rotate?-list)) :copy-worldcoords)))
             (tgt-cds (make-cascoords :coords (send obj-cds :copy-worldcoords)))
             (rotate-gripper (cadr (car final-cds-rotate?-list)))
             pre-av)
        (if rotate-gripper (send tgt-cds :rotate (deg2rad 180) :x))
        (send obj-cds :assoc tgt-cds)
        
        (unless (setq pre-av (send *pr2* :inverse-kinematics (send (send tgt-cds :copy-worldcoords) :translate #f(0 0 100) :world) ;;prepare cds (higher than tgt-cds)
                                   :move-target (send *pr2* arm :end-coords) :link-list (send *pr2* :link-list (send *pr2* arm :end-coords :parent)) :use-torso 0 :rotation-axis t :debug-view nil :warnp nil))
          (unless (setq pre-av (send *pr2* :inverse-kinematics (send (send tgt-cds :copy-worldcoords) :translate #f(0 0 50) :world) ;;prepare cds (higher than tgt-cds)
                                     :move-target (send *pr2* arm :end-coords) :link-list (send *pr2* :link-list (send *pr2* arm :end-coords :parent)) :use-torso 0 :rotation-axis t :debug-view nil :warnp nil)) (return-from push-boxes)))
        (push (send *pr2* :angle-vector) av-list)
        (print "point 0 ik-solved")
        
        (unless (send *pr2* :inverse-kinematics tgt-cds ;;target-cds
                      :move-target (send *pr2* arm :end-coords) :link-list (send *pr2* :link-list (send *pr2* arm :end-coords :parent)) :use-torso 0 :rotation-axis t :debug-view nil :warnp nil) (return-from push-boxes))
        (push (send *pr2* :angle-vector) av-list)
        (print "point 1 ik-solved")
        
        (send obj-cds :translate (if (eq *push-style* :use-palm) #f(0 0 100) #f(100 0 0)))
        (unless (send *pr2* :inverse-kinematics tgt-cds ;;after-push-cds
                      :move-target (send *pr2* arm :end-coords) :link-list (send *pr2* :link-list (send *pr2* arm :end-coords :parent)) :use-torso 0 :rotation-axis t :debug-view nil :warnp nil) (return-from push-boxes))
        (push (send *pr2* :angle-vector) av-list)
        (print "point 2 ik-solved")
        
        
        ;;robot-start-moving
        (setq av-list (reverse av-list))
        (send *ri* :angle-vector (elt av-list 0) 5000)
        (send *ri* :wait-interpolation)
        (send *ri* :angle-vector (elt av-list 1) 1000)
        (send *ri* :wait-interpolation)
        (send *ri* :angle-vector (elt av-list 2) 2000)
        (send *ri* :wait-interpolation)

        ;;finish pushing one object
        ;; (if pre-av
        ;;     (send *ri* :angle-vector pre-av 5000)
        (if (send *pr2* :larm :inverse-kinematics (send (send *pr2* :larm :end-coords :copy-worldcoords) :translate #f(0 0 200) :world))
            (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
          (if (send *pr2* :larm :inverse-kinematics (send (send *pr2* :larm :end-coords :copy-worldcoords) :translate #f(0 0 100) :world))
              (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
            (if pre-av
                (send *ri* :angle-vector pre-av 5000))))
        (send *ri* :wait-interpolation)
        
        ;;update box location
        (send target-box :translate (if (eq *push-style* :use-palm) #f(0 0 100) #f(100 0 0)) obj-cds)
        (setq *pushed-box* target-box)
        ))))




        
        
                    
       
         
        
  


  
      
          
  




    

    