
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import roboticstoolbox as rt
import rtde_receive
import rtde_control
from scipy.interpolate import griddata
# Orientation/rotation utility
from CSRL_math import *
from CSRL_orientation import *

from calculations import *
from spatialmath import  SE3, SO3
from scipy.linalg import block_diag
##############################################################################
# Keyboard Listener
##############################################################################
from pynput import keyboard
#from objective_function_with_kalman import *
from pinhole import *
import pyrealsense2 as rs
from pupil_apriltags import Detector 
import cv2
from current_extrinsic import *
from get_camera_position import *
from cmaes import CMAwM
import grasp_detector
import seaborn as sns
from scipy.special import gamma
import pandas as pd
import os

output_dir = "/home/csrl/Desktop/ROMAN_experiment/Outp"
os.makedirs(output_dir, exist_ok=True)

global global_counter,rewrd
global_counter =0
global currentParams
rewrd=0

keypoint_detections_times=[]
optimization_time=[]
loop_time =[]
stop_signal = False
def on_press(key):
    global stop_signal
    try:
        if key.char == 'a':
            print("Stopping robot...")
            stop_signal = True
            return False
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

detector = grasp_detector.TomatoDetector(model_path='models/keypoints_new.pt', show_frame=False)
#detector.run()


global P,KALMAN_GAIN,H,Q,F, frame



P = np.eye(18)
KALMAN_GAIN = np.zeros((18,18))
H = np.eye(18)
Q =  np.eye(18)*10
F = np.eye(18)
##############################################################################
#  objective_function
##############################################################################

z = np.zeros(18)
measurement_uncertainty  = np.eye(18)


def ellipsoid_volume():
    """Computes volume of an n-dimensional ellipsoid given covariance matrix P"""
    global P
    n = P.shape[0]  # Dimension (should be 18 in this case)
    det_P = np.linalg.det(P)

    # Ensure determinant is positive to avoid numerical errors
    if det_P <= 0:
        return 0

    volume = (np.pi ** (n / 2) / gamma(n / 2 + 1)) * np.sqrt(det_P)
    return volume


def objective_function_with_kalman(params):
    
    global P,KALMAN_GAIN,H,Q,F,rewrd
    
    phi, theta = params
    # Parameter constraints
    if not (np.pi/8< phi <= np.pi/2-0.3) or not (np.pi/8< theta <=  np.pi/2-0.3):
        return -np.inf  # Penalize invalid parameters
    
    try:
        cur_phi=phi
        cur_theta = theta
        
        
        q_ = np.array(rtde_r.getActualQ())
        g_ = UR_robot.fkine(q_)
        p_ = g_.t
        Ri = g_.R
        g0e = np.identity(4)
        g0e[0:3,0:3] = Ri.copy()
        g0e[0:3,3] = p_.copy()
        # Transform the 3D point to world coordinates using g0c
        g0c = g0e@gec


        #z,measurement_uncertainty=give_me_detections(cur_phi,cur_theta)
        keypoints = detector.get_last_keypoint_set()  # Retrieve the latest keypoint set
        
        if keypoints:
            for idx, kp in keypoints.items():
                p_te =  g0c @ np.append(kp["point"], 1)
                idx = int(idx)
                
                z[idx] = p_te[0]         # This gives you the 3D coordinates, e.g., [x, y, z]
                z[idx+6] = p_te[1]         # This gives you the 3D coordinates, e.g., [x, y, z]
                z[idx+12] = p_te[2]         # This gives you the 3D coordinates, e.g., [x, y, z]

                # # Find the diagonal position to update
                # diag_index = idx * 6  # For idx=0, it's the first 3 positions; for idx=1, the next 3 positions, etc.
                # # Fill the corresponding diagonal block with the keypoint's confidence
                # np.fill_diagonal(measurement_uncertainty[diag_index:diag_index+6, diag_index:diag_index+6], 1-kp["confidence"])
                # Compute and fill the correct diagonal positions
                diag_indices = [idx, idx + 6, idx + 12]
                # np.fill_diagonal(measurement_uncertainty[diag_indices, diag_indices], 1 - kp["confidence"])
                measurement_uncertainty[diag_indices, diag_indices] = 1 - kp["confidence"]
                # print(measurement_uncertainty)
                #print(f'idx: {idx}, confidence: {kp["confidence"]}')


                #print(f"Keypoint {idx}: Coordinates = {point}, Confidence = {confidence}")


        else:
            print("No keypoints detected in the current frame.")
        # print(f"Z = {z}")
        # print(f"measurement: {measurement_uncertainty}")


        iterat = 1
        if iterat == 1:
            x = np.array([-0.02, -0.02, -0.01,-0.025, -0.026, -0.018, -0.61, -0.68,-0.67,-0.57, -0.62,-0.65, 0.38,0.4,0.47,0.4,0.45,0.49]).T
            
        # # Set camera position
        
        # X_cam = p_des[0]
        # Y_cam = p_des[1]
        # Z_cam =  p_des[2]
        #print('APOSTASH',(X_cam-pivot_x)**2+(Y_cam-pivot_y)**2+(Z_cam-pivot_z)**2-radius**2)
        # Update camera extrinsics or other parapi/2meters as needed
        # c_extr = current_extrinsic(X_cam, Y_cam, Z_cam, pivot_x, pivot_y, pivot_z)

        # Initialize variables
        # x_hat = np.array([-0.77, -0.77, -0.77, -0.24, -0.44, -0.34,0.34,0.34,0.5]).T
        x_hat = x.copy()

        projected_point_0 = np.linalg.inv(T_des) @ np.hstack((x_hat[[0, 6, 12]],[1]))
        projected_point_1 = np.linalg.inv(T_des) @ np.hstack((x_hat[[1, 7, 13]],[1]))
        projected_point_2 = np.linalg.inv(T_des) @ np.hstack((x_hat[[2, 8, 14]],[1]))

        projected_point_3 = np.linalg.inv(T_des) @ np.hstack((x_hat[[3, 9, 15]],[1]))
        projected_point_4 = np.linalg.inv(T_des) @ np.hstack((x_hat[[4, 10, 16]],[1]))
        projected_point_5 = np.linalg.inv(T_des) @ np.hstack((x_hat[[5, 11, 17]],[1]))

        # Project the 3D point onto the 2D pixel plane
        pixelCoordstag0 = K @ projected_point_0[0:3]
        pixelCoordstag1 = K @ projected_point_1[0:3]
        pixelCoordstag2 = K @ projected_point_2[0:3]

        pixelCoordstag3 = K @ projected_point_3[0:3]
        pixelCoordstag4 = K @ projected_point_4[0:3]
        pixelCoordstag5 = K @ projected_point_5[0:3]

        if pixelCoordstag0[2] >= 0:
            # Normalize to get pixel coordinates
            pixelCoordstag0 = pixelCoordstag0 / pixelCoordstag0[2]
            utag0 = pixelCoordstag0[0]
            vtag0 = pixelCoordstag0[1]

        if pixelCoordstag1[2] >= 0:
            # Normalize to get pixel coordinates
            pixelCoordstag1 = pixelCoordstag1 / pixelCoordstag1[2]
            utag1 = pixelCoordstag1[0]
            vtag1 = pixelCoordstag1[1]

        if pixelCoordstag2[2] >= 0:
            # Normalize to get pixel coordinates
            pixelCoordstag2 = pixelCoordstag2 / pixelCoordstag2[2]
            utag2 = pixelCoordstag2[0]
            vtag2 = pixelCoordstag2[1]

        if pixelCoordstag3[2] >= 0:
            # Normalize to get pixel coordinates
            pixelCoordstag3 = pixelCoordstag3 / pixelCoordstag3[2]
            utag3 = pixelCoordstag3[0]
            vtag3 = pixelCoordstag3[1]

        if pixelCoordstag4[2] >= 0:
            # Normalize to get pixel coordinates
            pixelCoordstag4 = pixelCoordstag4 / pixelCoordstag4[2]
            utag4 = pixelCoordstag4[0]
            vtag4 = pixelCoordstag4[1]

        if pixelCoordstag5[2] >= 0:
            # Normalize to get pixel coordinates
            pixelCoordstag5 = pixelCoordstag5 / pixelCoordstag5[2]
            utag5 = pixelCoordstag5[0]
            vtag5 = pixelCoordstag5[1]

        M = np.eye(18)    
        # Check if the point is within the image boundaries
        if 0 <= utag0 <= image_width and 0 <= vtag0 <= image_height:
            SIGMA = measurement_uncertainty 
            # print(1)
        else:
            M[[0, 6, 12], [0, 6, 12]] = 10000000000000
            SIGMA = measurement_uncertainty @ M 
            # print(2)

        if 0 <= utag1 <= image_width and 0 <= vtag1 <= image_height:
            SIGMA = measurement_uncertainty 
            # print(3)
        else:
            M[[1, 7, 13],[1, 7, 13]] = 10000000000000
            SIGMA = measurement_uncertainty @ M 
            # print(4)

        if 0 <= utag2 <= image_width and 0 <= vtag2 <= image_height:
            SIGMA = measurement_uncertainty 
            # print(5)
        else:
            M[[2,8,14],[2,8,14]] = 10000000000000
            SIGMA = measurement_uncertainty @ M  
            # print(6)
        
        if 0 <= utag3 <= image_width and 0 <= vtag3 <= image_height:
            SIGMA = measurement_uncertainty 
            # print(1)
        else:
            M[[3, 9, 15], [3, 9, 15]] = 10000000000000
            SIGMA = measurement_uncertainty @ M 
            # print(2)

        if 0 <= utag4 <= image_width and 0 <= vtag4 <= image_height:
            SIGMA = measurement_uncertainty 
            # print(3)
        else:
            M[[4, 10, 16],[4, 10, 16]] = 10000000000000
            SIGMA = measurement_uncertainty @ M 
            # print(4)

        if 0 <= utag5 <= image_width and 0 <= vtag5 <= image_height:
            SIGMA = measurement_uncertainty 
            # print(5)
        else:
            M[[5,11,17],[5,11,17]] = 10000000000000
            SIGMA = measurement_uncertainty @ M  

        if iterat == 1:
            P = Q.copy()

        # PREDICT STAGE
        x_pred = F @ x 
        P_pred = F @ P @ F.T + Q

        # Kalman Filter Update
        y = z - H @ x_pred                  # Innovation

        bl_diag = block_diag(T_des[0:3, 0:3], T_des[0:3, 0:3], T_des[0:3, 0:3],T_des[0:3, 0:3], T_des[0:3, 0:3], T_des[0:3, 0:3])
        S = H @ P_pred @ H.T + bl_diag @ SIGMA @ bl_diag.T
        

        KALMAN_GAIN = P_pred @ H.T @ np.linalg.inv(S) # Kalman gain

        x = x_pred + KALMAN_GAIN @ y                  # Updated state estimate
        P = (np.eye(18) - KALMAN_GAIN @ H) @ P_pred    # Updated estimate covariance
        
        iterat += 1

        sign1, logdetP_pred = np.linalg.slogdet(P_pred)
        sign2, logdetP_post = np.linalg.slogdet(P)
        
        # if sign1 <= 0 or sign2 <= 0:
        #     rewrd = 1e5
        # else:
        # info_gain = abs(0.5*(logdetP_pred-logdetP_post))
        
        det_prior = np.linalg.det(P_pred)
        det_post = np.linalg.det(P)
        
        
        # division-based log form
        #info_gain = abs(np.log(det_prior/det_post))#because cmaes minimizes
        info_gain = np.log((det_prior/det_post))#because cmaes minimizes
        rewrd = info_gain
        # rewrd= np.linalg.det(P)
        return rewrd

    except Exception as e:
        print(f"Exception in objective function 3: {e}")
        return -np.inf 

    


##############################################################################
# 1) look_at_center => orientation
##############################################################################
def look_at_center(p_current, p_center, gravity_vec= np.array([0, 0, 1])):
    
    z_vec = p_center - p_current
    norm_z = np.linalg.norm(z_vec)
    forward = z_vec / norm_z # Normalize forward vector
    
    # Up vector (towards z-axis)
    up = -np.array([0, 0, 1])
    # up = np.array([0, 0, 1])
    # Check for the degenerate case at the "north pole" (Phi = 0, forward = up)
    if np.linalg.norm(forward - up) < 1e-6:
        # When forward and up are the same, we set an arbitrary right vector
        right = np.array([1, 0, 0])  # You can choose any vector perpendicular to up
        up = np.cross(right, forward)  # Recalculate up as orthogonal to forward and right
    else:
        # Right vector (cross product of forward and up)
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)  # Normalize right vector

        # Recompute up vector as orthogonal to forward and right
        up = np.cross(forward, right)

    R_des = np.column_stack((right, up, forward))
    return R_des


##############################################################################
# 5) Look-at-Center Orientation with Feedforward
##############################################################################
def orientation_look_at_center(t):
    """
    Compute R_des(t) and w_des(t) using SLERP with a quintic profile for alpha(t),
    dynamically ensuring the z-axis always points toward the hemisphere center.

    Args:
        t (float): Current time.

    Returns:
        R_des (np.ndarray): Desired 3x3 rotation matrix.
        w_des (np.ndarray): Desired angular velocity (3D vector).
    """

    p_des, v_des = polar_position_and_velocity(t)

    # Orientation: z-axis points to center
    R_des = look_at_center(p_des, p_center)

    r =   p_des-p_center 
    
    w_des = np.cross(r,v_des) / (radius * radius)
    
    return R_des, w_des


##############################################################################
# 2) Quintic alpha(t), alpha_dot(t)
##############################################################################
def alpha_profile(t):
    """
    Returns alpha(t) and alpha_dot(t) in [0,1] using quintic from 0->1 over 'duration'.
    """
    s = np.clip(t / duration, 0.0, 1.0)
    alpha = 10*s**3 - 15*s**4 + 6*s**5
    # derivative wrt s: alpha'(s) = 30 s^2 - 60 s^3 + 30 s^4
    alpha_sdot = 30*s**2 - 60*s**3 + 30*s**4
    # ds/dt = 1/duration
    alpha_tdot = alpha_sdot / duration
    return alpha, alpha_tdot

##############################################################################
# 3) Position on Hemisphere + derivative
##############################################################################
def polar_position_and_velocity(t):
    global currentParams
    """
    Returns:
      p_des(t), v_des(t)
    where p_des(t) is on the hemisphere, and v_des(t) = dp_des/dt
    using polar coords (phi(t),theta(t)) each driven by alpha(t).
    """
    alpha, alpha_dot = alpha_profile(t)
    # polar angles
    phi_t   = phi0   + alpha*(phi1   - phi0)
    theta_t = theta0 + alpha*(theta1 - theta0)
    # derivatives
    phi_dot   = (phi1 - phi0)*alpha_dot
    theta_dot = (theta1 - theta0)*alpha_dot


    sinphi = math.sin(phi_t)
    cosphi = math.cos(phi_t)
    sintheta = math.sin(theta_t)
    costheta = math.cos(theta_t)

    x = radius*sinphi*costheta
    y = radius*sinphi*sintheta
    z = radius*cosphi
    p_des = p_center + np.array([x,y,z])

    # partial derivatives => dp/dt
    # dx/dphi = r cosphi costheta, dx/dtheta = -r sinphi sintheta
    dx_dphi = radius*( cosphi*costheta )
    dx_dtheta = radius*( -sinphi*sintheta )
    # dy/dphi = r cosphi sintheta, dy/dtheta = r sinphi costheta
    dy_dphi = radius*( cosphi*sintheta )
    dy_dtheta= radius*( sinphi*costheta )
    # dz/dphi = -r sinphi,        dz/dtheta= 0
    dz_dphi = radius*( -sinphi )
    dz_dtheta= 0.0

    # chain rule => dp/dt = partial wrt phi * phi_dot + partial wrt theta * theta_dot
    vx = dx_dphi*phi_dot + dx_dtheta*theta_dot
    vy = dy_dphi*phi_dot + dy_dtheta*theta_dot
    vz = dz_dphi*phi_dot + dz_dtheta*theta_dot
    v_des = np.array([vx, vy, vz])

    currentParams=[phi_t,theta_t]#---------------------------------------------------------------------------
    return p_des, v_des

def desired_pose_polar_with_look_at(t):
    """
    Returns:
      - T_des: 4x4 transform with p_des(t) + R_des(t)
      - v_des: 3D linear velocity
      - w_des: 3D angular velocity
    """
    
    p_des, v_des = polar_position_and_velocity(t)

    
    R_des, w_des = orientation_look_at_center(t)
    # print(is_rotation_matrix(R_des,tol=1e-6))#!!!!!!
    # Build homogeneous transform
    T_des = np.eye(4)
    T_des[:3, :3] = R_des
    T_des[:3, 3] = p_des

    
    #T_des = T_des@gce

    return T_des, v_des, w_des


def margin_to_uncertainty(margin, low, high):
    """Example heuristic that clamps decision_margin between low & high."""
    if margin < low:
        return 0.0
    elif margin > high:
        return 1.0
    else:
        return ((margin - low) / (high - low))


def give_me_detections(evaluated_phi,evaluated_theta):
    global frame
    exit_flag = False
    
    # Define target tag IDs
    target_tags = [0, 1, 2]

    while not exit_flag:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue  # Skip frame if either stream is unavailable

        # Convert color frame to numpy array and grayscale
        frame = np.asanyarray(color_frame.get_data())
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_height, frame_width = frame_gray.shape

        # Detect AprilTags in the grayscale image
        detected_tags = {}  # Will store detected centers: {tag_id: (center_x, center_y)}
        tags = detector.detect(frame_gray)
        
        for tag in tags:
            if tag.tag_id in target_tags:
                # Compute the center of the tag from its corners
                center = np.mean(tag.corners, axis=0).astype(int)
                center_x, center_y = int(center[0]), int(center[1])
                detected_tags[tag.tag_id] = ((center_x, center_y), tag.decision_margin)
            
                
                # Draw the bounding box (green lines) around the detected tag
                for idx in range(len(tag.corners)):
                    pt1 = tuple(tag.corners[idx - 1, :].astype(int))
                    pt2 = tuple(tag.corners[idx, :].astype(int))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                # Mark the center in red
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # Prepare a dictionary to hold information for each target tag
        tag_positions = {}
        
        # For each target tag, use detected center if available; otherwise assign a random center.
        for tag_id in target_tags:
            if tag_id in detected_tags:
                (center_x, center_y), decision_margin = detected_tags[tag_id]
            else:
                # Assign a random center within the frame dimensions
                center_x = np.random.randint(0, frame_width)
                center_y = np.random.randint(0, frame_height)
                decision_margin = 0        #!!!!-----------------high uncertainty
                # Mark the random center in blue so you know it wasn't detected
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)


            decision_margin=margin_to_uncertainty(decision_margin, low=0, high=100)
            # Get the depth value (in meters) at the chosen center coordinate
            depth_value = depth_frame.get_distance(center_x, center_y)

            if depth_value>4.0 or depth_value<0 or math.isnan(depth_value):
                depth_value = 0.4

            # Deproject the pixel to a 3D point in the camera coordinate system
            p_camera = np.ones(4)
            p_camera[0:3] = ph.back_project([center_x, center_y], depth_value)
            q_ = np.array(rtde_r.getActualQ())
            g_ = UR_robot.fkine(q_)
            p_ = g_.t
            Ri = g_.R
            g0e = np.identity(4)
            g0e[0:3,0:3] = Ri.copy()
            g0e[0:3,3] = p_.copy()
            # Transform the 3D point to world coordinates using g0c
            g0c = g0e@gec
            p_te = g0c @ p_camera
            p_world = p_te[0:3]
            
            ###--------------------------------------------Deproject the pixel sol2-------------------------------

            # color_profile = profile.get_stream(rs.stream.color)
            # depth_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

            # # Deproject the pixel to a 3D point in camera coordinates
            # p_camera = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [center_x, center_y], depth_value)

            # # Convert to homogeneous coordinates for transformation (assuming g0c is defined)
            # p_camera_h = np.append(p_camera, 1)  # [x, y, z, 1]


            # q_ = np.array(rtde_r.getActualQ())
            # g_ = UR_robot.fkine(q_)
            # p_ = g_.t
            # Rini = g_.R
            # g0e = np.identity(4)
            # g0e[0:3,0:3] = Rini.copy()
            # g0e[0:3,3] = p_.copy()

            # g0c =  g0e @ gec#######################!!!!!!!!!!!!!!!!-----
            # p_world = np.dot(g0c, p_camera_h)[:3]  # -------------------get the centers of APRILTAGS with respect to WORLD FRAME!!!!!
            ###--sol2--------

            # Save the computed information for this tag
            tag_positions[tag_id] = {
                'center': (center_x, center_y),
                'depth': depth_value,
                'p_camera': p_camera,
                'p_world': p_world,
                'decision_margin': decision_margin
            }
            
           
            # cv2.imshow(f"PHI,THETA: {evaluated_phi,evaluated_theta}", frame)
            # key = cv2.waitKey(1)

            # if key == 27:  # Exit loop when 'Esc' key is pressed
            #     break
            if depth_value<2 and depth_value>0.2:
                exit_flag = True
                
            
    # State Space Matrix
    uncertainty_apritag0 = tag_positions[0]['decision_margin']
    uncertainty_apritag1 = tag_positions[1]['decision_margin']
    uncertainty_apritag2 = tag_positions[2]['decision_margin']

    p_world_apritag0 = tag_positions[0]['p_world']
    p_world_apritag1 = tag_positions[1]['p_world']
    p_world_apritag2 = tag_positions[2]['p_world']
    POSIT = np.array([p_world_apritag0[0], p_world_apritag0[1], p_world_apritag0[2],p_world_apritag1[0], p_world_apritag1[1], p_world_apritag1[2],p_world_apritag2[0], p_world_apritag2[1], p_world_apritag2[2]]).T
    Uncert = np.diag(np.array([uncertainty_apritag0, uncertainty_apritag1, uncertainty_apritag2, uncertainty_apritag0, uncertainty_apritag1, uncertainty_apritag2,uncertainty_apritag0, uncertainty_apritag1, uncertainty_apritag2]))
    return POSIT, Uncert

##############################################################################
# MAIN SCRIPT
##############################################################################
global p_center, radius, phi0, phi1, theta0, theta1, duration

#p_center = np.array([-0.15, -0.54, 0.60])  # center of hemisphere!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
p_center = np.array([-0.030330967842613382, -0.6587281285231606, 0.4205417579406674])
#p_center = np.array([-0.288, -0.828, 0.537]) ###to kentro symmetriaqs twn apriltags
# pivot_x =p_center[0]
# pivot_y = p_center[1]
# pivot_z = p_center[2]
p_center_x =p_center[0]
p_center_y = p_center[1]
p_center_z = p_center[2]
radius   = 0.25

# ------------------Initialize RealSense D415 pipeline------------------------------

# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# # Create an align object (aligns depth to color stream)
# align = rs.align(rs.stream.color)
# # Start the pipeline
# profile = pipeline.start(config)

# AprilTag Detector Initialization
# detector = Detector(families="tag36h11")
# color_stream_profile = profile.get_stream(rs.stream.color)
# video_profile = color_stream_profile.as_video_stream_profile()
# intrinsics = video_profile.get_intrinsics()  # rs.intrinsics structure


Rec = np.identity(3)
# Rec[0,0] = -1
# Rec[1,1] = -1
#pec = np.array([-0.04, -0.015, 0.08])
pec = np.array([-0.035, -0.015, 0.08])

gec = np.identity(4)
gec[0:3,0:3] = Rec.copy()
gec[0:3,3] = pec.copy()

gce = np.linalg.inv(gec)


# Camera intrinsic parameters 

# fx = intrinsics.fx  # Focal length in x
# fy =  intrinsics.fy  # Focal length in y
# cx = intrinsics.ppx # Principal point x (center of the image)
# cy = intrinsics.ppy  # Principal point y (center of the image)

# # Camera intrinsic matrix - K
# K = np.array([[fx, 0, cx],
#               [0, fy, cy],
#               [0, 0, 1]])

K= np.array([[633.5527954101562, 0, 640.886474609375],
            [0, 633.5527954101562, 363.0577087402344],
            [0, 0, 1]])

# # 
# image_width = intrinsics.width
# image_height = intrinsics.height
image_width = 640
image_height = 480

# ph = PinholeCamera(fx, fy, cx, cy, image_width, image_height)


if __name__ == "__main__":
    partial_reward=0#-------------------------------------------------------
    # A) RTDE
    robot_ip = "192.168.1.100"
    rtde_c = rtde_control.RTDEControlInterface(robot_ip)
    rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    tcp_offset = [-0.04, -0.015, 0.08, 0, 0, 0]
    # rtde_c.setTcp(tcp_offset)
    # tcp_offset = [0, 0, 0, 0, 0, 0]
    rtde_c.setTcp(tcp_offset)
    # B) UR Robot Model
    pi = math.pi
    UR_robot = rt.DHRobot([
        rt.RevoluteDH(d=0.1625, alpha=pi/2),
        rt.RevoluteDH(a=-0.425),
        rt.RevoluteDH(a=-0.3922),
        rt.RevoluteDH(d=0.1333, alpha=pi/2),
        rt.RevoluteDH(d=0.0997, alpha=-pi/2),
        rt.RevoluteDH(d=0.0996)
    ], name='UR5e')
    

    #tcp_offset_position = [-0.04, -0.015, 0.08] #D415
    tcp_offset_position = [-0.035, -0.015, 0.08] #D435
    # Define rotation (axis-angle format)
    rotation_axis = np.array([0, 0, 0])  # Rotation vector
    rotation_angle = np.linalg.norm(rotation_axis)  # Angle in radians
    rotation_unit_vector = rotation_axis / rotation_angle if rotation_angle != 0 else rotation_axis  # Normalize

    # Convert to rotation matrix
    rotation_matrix = SO3.AngVec(rotation_angle, rotation_unit_vector)

    # Create the SE(3) transformation
    TCP_offset = SE3.Trans(*tcp_offset_position) * SE3(rotation_matrix)

    # Set TCP on UR5e model
    UR_robot.tool = TCP_offset

    # C) Move to an Initial Joint Configuration ------ HOME POSITION
    q0_deg = [81.7, -91.43, 91.18, -148.57, -124.27, 24]

    q0_rad = np.radians(q0_deg)
    rtde_c.moveJ(q0_rad, 0.5, 0.5)


    q_init = np.array(rtde_r.getActualQ())
    g_init = UR_robot.fkine(q_init)
    p_init_robot = g_init.t
    R_init_robot = g_init.R

    # R_des = R_init_robot

    
    g0e = np.identity(4)
    g0e[0:3,0:3] = R_init_robot.copy()
    g0e[0:3,3] = p_init_robot.copy()

    g0c =  g0e @ gec#######################!!!!!!!!!!!!!!!!-----


    #___MEXRI EDW PAEI SE MIA TYXAIA ARXIKH THESH


    phi0   = np.pi/6
    
    theta0 = np.pi/3
  
    dt        = 0.02     # 20 ms ()
    seg_time  =5     # time (sec) for point to point movement with quintic (5th order) profile

    time_now  = 0.0
    duration = seg_time #seconds

    kClik = np.identity(6)

    kClik_gain = 2
    kClik[0:3, 0:3] = (kClik_gain)* kClik[0:3, 0:3]
    kClik[3:6, 3:6] =  kClik_gain* kClik[3:6, 3:6]
    
    

    # Logging
    time_log = []
    pos_log = []
    pos_des_log = []
    ori_log = []
    ori_des_log = []

    # F) Control Loop
    t = 0.0
    # 3) 6D Error
    e = np.zeros(6)

######################################################**********************************#############################################
######################################################**********************************#############################################


    #measurements = [z0, z1, z2]
   
    # Define ranges for phi and theta for different starting points of camera!!!!

    # phi_range = np.linspace(np.pi / 5, np.pi / 3, 7)   
    # theta_range = np.linspace(0.0, 2 * np.pi - (1/5)* 2 * np.pi, 7)
    phi_range = np.linspace(np.pi/8+0.001,np.pi/2-0.3-0.001, 15)   
    theta_range = np.linspace(np.pi/8+0.001,np.pi/2-0.3-0.001, 15)
    phi_grid, theta_grid = np.meshgrid(phi_range, theta_range)

    phi_combinations = phi_grid.flatten()
    theta_combinations = theta_grid.flatten()

    # CMA-ES Optimization with Multiple Experiments


    input('When UR ready :)...please press "Enter"...')

    
    initial_parameters = np.array([phi0,theta0])
    cma_bounds = np.array([[np.pi/8,np.pi/2-0.3], [np.pi/8,np.pi/2-0.3]])
    steps = np.ones(2)*0.001  # Continuous dimensions only

    population_size =5
    optimizer = CMAwM(
        mean=initial_parameters,
        sigma=2,#1000.0,
        bounds=cma_bounds,
        steps=steps,
        population_size=population_size
    )

    cma_points = [initial_parameters]
    all_desired_positions = [p_init_robot]
    best_experiment_value = -np.Inf
    best_experiment_params = None
    num_of_experiments = len(phi_combinations)
    best_results =[]
    step_count = 0

    visualize=[]
    volume = []  # List to store sigma over iterations

    for exper in range(num_of_experiments):
        

        if (not stop_signal):
            # 1) Ζητάς μια νέα υποψήφια λύση από το CMA-ES
            
            
            phi1 = phi_combinations[exper]
            theta1 = theta_combinations[exper]
    
            
            # Reset simulation time, if applicable
            t_local = 0.0
            t_start = rtde_c.initPeriod()
            while t_local < seg_time and not stop_signal:
            
                time_start = time.time()
                t = t_local
                #t += dt
                # 1) Current Robot State
                q = np.array(rtde_r.getActualQ())
                J = UR_robot.jacob0(q)
                Jinv = np.linalg.pinv(J)
                g_current = UR_robot.fkine(q)
                p_temp = g_current.t
                R_current = g_current.R

                # 2) Desired Pose with Look-at-Center Orientation
                T_des, v_des, w_des = desired_pose_polar_with_look_at(t)
                p_des = T_des[:3, 3]
                R_des = quatIntegrate(T_des[:3, :3], w_des, dt)
                # R_des = T_des[:3, :3]


                # 3) 6D Error
                
                e[:3] = p_des - p_temp
                e[3:] = logError(R_des, R_current, getMin=True)

                
                # 4) Command (6D feedforward - error correction)
                # correction = kClik @ e
                correction = np.concatenate([v_des, w_des]) + kClik @ e

                # 5) joint velocity
                qdot = Jinv @ correction

                # 6) command speed
                rtde_c.speedJ(qdot, 1.0, dt)
            

                # (vi) Partial cost (Kalman + detections)
                optimization_time_start=time.time()
                partial_reward = objective_function_with_kalman(currentParams)######### edw thelei allagh.....8a prepei na einai to current phi,theta
                optimization_time.append(time.time()-optimization_time_start)
                # print("t_local =",t_local)
                visualize.append((currentParams, partial_reward))
                 
                if partial_reward >best_experiment_value:
                    best_experiment_value = partial_reward
                    best_experiment_params = currentParams
                    
                    best_results.append((currentParams, partial_reward)) 
                    filename = os.path.join(output_dir, f"PHITHETA_{best_experiment_params[0],best_experiment_params[1],partial_reward}.png")
                    cv2.imwrite(filename, detector.get_last_frame())


                if stop_signal:
                    break
                rtde_c.waitPeriod(dt)
                t_local   += dt
                #rtde_c.waitPeriod(dt)
                time_end = time.time()
                elapsed_time = time_end-time_start
                remaining_time = dt -elapsed_time

                if remaining_time > 0:
                    time.sleep(dt - elapsed_time)
                    dt_time = time.time() - time_start
                    # print("dt_time", dt_time)
                else:
                    dt_time = time.time() - time_start
                    # print("dt_time", dt_time)
                    
                loop_time.append(elapsed_time)

            #volume.append(ellipsoid_volume())
            #visualize.append((currentParams, partial_reward))
            
            phi0 = currentParams[0]
            theta0 = currentParams[1]
            print("Progress (%): ",(exper/num_of_experiments)*100)
              
    #print(visualize)        
    print("Progress (%): ",100)    
    t_local = 0.0
    rtde_c.speedStop()
    input("Press 'Enter' gia Na Paw Sthn Optimal Pose")
    best_params, best_reward = max(best_results, key=lambda x: x[1])
    # print(best_params)
    phi1 = best_params[0]
    theta1 = best_params[1]
    # print("Final - Best Position:", phi1, theta1)  
    time_start = time.time()
    while t_local < seg_time and not stop_signal:
        time_start = time.time()
        t_start = rtde_c.initPeriod()
        t = t_local

        # 1) Current Robot State
        q = np.array(rtde_r.getActualQ())
        J = UR_robot.jacob0(q)
        Jinv = np.linalg.pinv(J)
        g_current = UR_robot.fkine(q)
        p_temp = g_current.t
        R_current = g_current.R

        # 2) Desired Pose with Look-at-Center Orientation
        T_des, v_des, w_des = desired_pose_polar_with_look_at(t)
        p_des = T_des[:3, 3]
        R_des = quatIntegrate(T_des[:3, :3], w_des, dt)
        # R_des = T_des[:3, :3]


        # 3) 6D Error
        
        e[:3] = p_des - p_temp
        e[3:] = logError(R_des, R_current, getMin=True)

        # 4) Command (6D feedforward - error correction)
        # correction = kClik @ e
        correction = np.concatenate([v_des, w_des]) + kClik @ e

        # 5) joint velocity
        qdot = Jinv @ correction

        # 6) command speed
        rtde_c.speedJ(qdot, 1.0, dt)
        rtde_c.waitPeriod(dt)
        t_local += dt
        
        time_end = time.time()
        elapsed_time = time_end-time_start
        remaining_time = dt -elapsed_time

        if remaining_time > 0:
            time.sleep(dt - elapsed_time)
            dt_time = time.time() - time_start
            # print("dt_time", dt_time)
        else:
            dt_time = time.time() - time_start
    #end_time = time.time() 
    #print(e)   
    #ellapsed_time = end_time - time_start
    # Τέλος
    print("Finish")
    #print(f"time: {ellapsed_time}")
    rtde_c.speedStop()
  

# Plot all three time series

# plt.plot(optimization_time, label="optimization_time", linewidth=2)
# plt.plot(detector.get_process_time(), label="detection time", linewidth=2)
# plt.plot(loop_time, label="loop time", linewidth=2)

# # Add legend
# plt.legend()

# # Show grid
# plt.grid(True)

# # Show the plot
# plt.show()
######################################################**********************************#############################################
######################################################**********************************#############################################
###



# Convert to DataFrame
# df = pd.DataFrame(data, columns=['X', 'Y', 'Reward'])

# # Aggregate duplicate (X, Y) values by averaging rewards (change to max/min if needed)
# df = df.groupby(['X', 'Y'], as_index=False)['Reward'].mean()

# df['Reward'].replace([-np.inf], 0, inplace=True)

# # Pivot table for heatmap format
# heatmap_data = df.pivot(index='X', columns='Y', values='Reward')

# # Convert index and column names to formatted strings to ensure precision in display
# heatmap_data.index = [f"{x:.4f}" for x in heatmap_data.index]
# heatmap_data.columns = [f"{y:.4f}" for y in heatmap_data.columns]

# # Plot using seaborn with colors only (no text inside)
# plt.figure(figsize=(10, 8))
# sns.heatmap(heatmap_data, annot=True, cmap='viridis', 
#             linewidths=0.5, cbar_kws={'label': 'Confidence Score'})

# plt.title("Reward Heatmap")
# plt.xlabel("Y-axis")
# plt.ylabel("X-axis")

# # Rotate axis labels for better readability
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)

# plt.show()




data = [(round(x, 4), round(y, 4), round(z, 4)) for ((x, y), z) in visualize]

# Convert to DataFrame
df = pd.DataFrame(data, columns=['X', 'Y', 'Reward'])
df['Reward'].replace([-np.inf], 0, inplace=True)
# Aggregate duplicate (X, Y) values by averaging rewards
#df = df.groupby(['X', 'Y'], as_index=False)['Reward'].mean()
# Identify initial and optimal points
initial_point = df.iloc[0]  # Assuming the first recorded point is the initial
optimal_point = df.loc[df['Reward'].idxmax()]  # Point with the highest reward
# Scatter Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df.X, df.Y, c=df.Reward, cmap='viridis', edgecolors='black', linewidth=0.5, s=50)

# Highlight initial and optimal points
plt.scatter(initial_point.X, initial_point.Y, color='red', marker='o', s=150, label="Initial Point", edgecolors='black', linewidth=1.5)
plt.scatter(optimal_point.X, optimal_point.Y, color='red', marker='*', s=200, label="Optimal Point", edgecolors='black', linewidth=1.5)

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label("Reward Intensity")

# Labels and Title
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Scatter Plot of Reward Intensity")
plt.legend()
plt.show()