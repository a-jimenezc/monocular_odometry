import numpy as np
import cv2
import math
import time 
import platform 


traj_img_size = 800
traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
half_traj_img_size = int(0.5*traj_img_size)

"""
    usage: feed it with poses list
"""
def drawTraj2d(poses_est:list,draw_scale=10):
    
    img_id = 0    
    for pose in poses_est:
        
        x, y, z = pose.t[0],pose.t[1],pose.t[2]
        draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
        cv2.circle(traj_img, (draw_x, draw_y), 1, (img_id*255/4540, 255-img_id*255/4540, 0), 1) 
        cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
        cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        
        cv2.imshow("Trajectory2d",traj_img)    
        cv2.waitKey(1)
        img_id += 1

        key_cv = cv2.waitKey(1) & 0xFF
        if  (key_cv == ord('q')): 
            break
            
    while True:
        key_cv = cv2.waitKey(1) & 0xFF
        if  (key_cv == ord('q')): 
            break
        
    cv2.destroyAllWindows()
