
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import roboticstoolbox as rt
import rtde_receive
import rtde_control

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

import pandas as pd
import os
from scipy.io import savemat

output_mat_dir = "/home/csrl/Desktop/ROMAN_experiment/mat_files"
os.makedirs(output_mat_dir, exist_ok=True)


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



global P,KALMAN_GAIN,H,Q,F, frame



P = np.eye(18)
KALMAN_GAIN = np.zeros((18,18))
H = np.eye(18)
Q =  np.eye(18)*10
F = np.eye(18)
##############################################################################
#  objective_function
##############################################################################

z = np.zeros((18,1))
x = np.zeros((18,1))
measurement_uncertainty  = np.eye(18)

def objective_function_with_kalman(params):
    
    global P,KALMAN_GAIN,H,Q,F,rewrd
    
    phi, theta = params
    # Parameter constraints
    if not (np.pi/8< phi <= np.pi/2-0.3) or not (np.pi/8< theta <=  np.pi/2-0.3):
        return -np.inf  # Penalize invalid parameters
    
    #try:
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
        x = np.array([-0.02, -0.02, -0.01,-0.025, -0.026, -0.018, -0.61, -0.68,-0.67,-0.57, -0.62,-0.65, 0.38,0.4,0.47,0.4,0.45,0.49]).reshape((18, 1))
        #x = np.array([-0.02, -0.02, -0.01,-0.025, -0.026, -0.018, -0.61, -0.68,-0.67,-0.57, -0.62,-0.65, 0.38,0.4,0.47,0.4,0.45,0.49])
        
    # Ensure that i am always to hemishere
    # X_cam = p_des[0]
    # Y_cam = p_des[1]
    # Z_cam =  p_des[2]
    # print('dist:',(X_cam-p_center[0])**2+(Y_cam-p_center[1])**2+(Z_cam-p_center[2])**2-radius**2)
    
    x_hat = x.copy()
    
    projected_point_0 = np.linalg.inv(T_des) @ np.vstack((x_hat[[0, 6, 12]],[1]))
    projected_point_1 = np.linalg.inv(T_des) @ np.vstack((x_hat[[1, 7, 13]],[1]))
    projected_point_2 = np.linalg.inv(T_des) @ np.vstack((x_hat[[2, 8, 14]],[1]))

    projected_point_3 = np.linalg.inv(T_des) @ np.vstack((x_hat[[3, 9, 15]],[1]))
    projected_point_4 = np.linalg.inv(T_des) @ np.vstack((x_hat[[4, 10, 16]],[1]))
    projected_point_5 = np.linalg.inv(T_des) @ np.vstack((x_hat[[5, 11, 17]],[1]))

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
    
    
    
    det_prior = np.linalg.det(P_pred)
    det_post = np.linalg.det(P)
    
    
    info_gain = np.log((det_prior/det_post))
    rewrd = info_gain 

    rewrd = rewrd #+ 100/(pow(pixelCoordstag0[0] - image_width/2 , 2) + pow(pixelCoordstag0[1] - image_height/2 , 2) )
    # rewrd= np.linalg.det(P)
    return rewrd

    # except Exception as e:
    #     print(f"Exception in objective function: {e}")
    #     return -np.inf 

    


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
        right = np.array([1, 0, 0])  # vector perpendicular to up
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
    
    # Build homogeneous transform
    T_des = np.eye(4)
    T_des[:3, :3] = R_des
    T_des[:3, 3] = p_des

    
    #T_des = T_des@gce

    return T_des, v_des, w_des


##############################################################################
# MAIN SCRIPT
##############################################################################
global p_center, radius, phi0, phi1, theta0, theta1, duration

#p_center = np.array([-0.15, -0.54, 0.60])  # center of hemisphere!!!!!!!
p_center = np.array([-0.030330967842613382, -0.6587281285231606, 0.4205417579406674])
#p_center = np.array([-0.288, -0.828, 0.537]) ###to kentro symmetriaqs twn apriltags
# pivot_x =p_center[0]
# pivot_y = p_center[1]
# pivot_z = p_center[2]
p_center_x =p_center[0]
p_center_y = p_center[1]
p_center_z = p_center[2]
radius   = 0.25


Rec = np.identity(3)

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

    # CMA-ES Optimization with Multiple Experiments
    for experiment_index in range(1):
        phi0   = np.pi/6
        theta0 = np.pi/3


        # phi0   = 0.8
        # theta0 = 0.5
        
        input('When UR ready :)...please press "Enter"...')

        

        #num_of_experiments = 1#len(phi_combinations)


        # for exper in range(num_of_experiments):
        initial_parameters = np.array([phi0,theta0])
        cma_bounds = np.array([[np.pi/8,np.pi/2-0.3], [np.pi/8,np.pi/2-0.3]])
        steps = 0.01*np.ones(2)  # 

        population_size =6
        optimizer = CMAwM(
            mean=initial_parameters,
            sigma=6,#2,
            bounds=cma_bounds,
            steps=steps,
            population_size=population_size
        )

        

        cma_points = [initial_parameters]
        all_desired_positions = [p_init_robot]
        best_experiment_value = -np.Inf
        best_experiment_params = None


        x_cur=initial_parameters

        # --- Optimization Loop ---
        # Prepare a list to store sigma values for visualization
        partial_reward_logs=[]
        phi1_logs =[]
        theta1_logs =[]
        phi_logs = []
        theta_logs =[]
        cmaes_sigma_log =[]
        covariance_log=[]
        visualize = []
        best_results =[]
        step_count = 0

        for i in range (5): #  5 epochs ( iterations)
            solutions = [] 
            solutions_for_cma_manipulation  = [] 
        # 7)  
            for its in range(population_size): #----------------
                if (not stop_signal):
                    # 1)  Ask CMAwM-ES for canditate solution
                    
                    x_for_eval, x_for_tell = optimizer.ask()
                    print(x_for_eval)
                    phi1 = x_for_eval[0]
                    theta1 = x_for_eval[1]
            
                    theta1_logs.append(theta1)
                    phi1_logs.append(phi1)

                    # Reset simulation time, 
                    t_local = 0.0
                    t_start = rtde_c.initPeriod()
                    while t_local < seg_time and not stop_signal:
                    
                        time_start = time.time()
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
                    

                        # (vi) Partial cost (Kalman + detections)
                        optimization_time_start=time.time()
                        partial_reward = objective_function_with_kalman(currentParams)#
                        optimization_time.append(time.time()-optimization_time_start)
                        # print("t_local =",t_local)

                        partial_reward_logs.append(partial_reward)

                        curr_phi, curr_theta = currentParams
                        phi_logs.append(curr_phi)
                        theta_logs.append(curr_theta)

                        visualize.append((currentParams, partial_reward))
                        if partial_reward >best_experiment_value:
                            best_experiment_value = partial_reward
                            best_experiment_params = currentParams
                            
                            best_results.append((currentParams, partial_reward)) 
                            filename = os.path.join(output_dir, f"PHITHETA_{best_experiment_params[0],best_experiment_params[1],partial_reward}.png")
                            cv2.imwrite(filename, detector.get_last_frame())


                        step_count += 1
                        if (step_count%50 == 0): #Append per....
                            #step_count += 1
                            solutions_for_cma_manipulation.append((currentParams, partial_reward)) 
                            
                        
                        if (len(solutions_for_cma_manipulation)==population_size): #Tell  (Update Optimizer) per...
                            
                            optimizer.tell(solutions_for_cma_manipulation)
                            solutions_for_cma_manipulation  = [] 
                            # Log the current sigma value
                            covariance_log.append(optimizer._cma._C.copy())
                            cmaes_sigma_log.append(optimizer._cma._sigma.copy())
                            
                            
                            #step_count = 0
                        
                    #
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

                

                    phi0 = currentParams[0]
                    theta0 = currentParams[1]
                    
                #optimizer.tell(solutions)    
        best_params, best_reward = max(best_results, key=lambda x: x[1])    
        # Prepare data to save for this experiment
        experiment_data = {
            'phi1': phi1_logs,
            'theta1': theta1_logs,
            'partial_rewards': partial_reward_logs,
            'cur_phi': phi_logs,
            'cur_theta': theta_logs,
            'cmaes_sigma_log': cmaes_sigma_log,
            'covariance_log': covariance_log,
            'best_results':best_params,
            'best_reward':best_reward
        }

         # Save each experiment's data to a separate .mat file
        mat_file_name = f"experimnt_{experiment_index + 1}.mat"
        savemat(os.path.join(output_mat_dir, mat_file_name), experiment_data)

        t_local = 0.0
        rtde_c.speedStop()
    
    input("Press 'Enter' to go to the Optimal Pose!")
    best_params, best_reward = max(best_results, key=lambda x: x[1])
    print("Best Params", best_params)
    print("Best_reward",best_reward)
    
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
  

######################################################**********************************#############################################
######################################################**********************************#############################################
###
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


det_values = [np.linalg.det(C) for C in covariance_log]

# Option A: Plot the raw determinant
plt.plot(det_values, label='det(Covariance)')
plt.xlabel('Iteration')
plt.ylabel('Determinant')
plt.title('Det(C) over Iterations')
plt.grid(True)
plt.legend()
plt.show()

sigma_values = [sigma for sigma in cmaes_sigma_log]

# Option A: Plot the raw determinant
plt.plot(sigma_values, label='sigma_values')
plt.xlabel('Iteration')
plt.ylabel('Sigma')
plt.title('sigma')
plt.grid(True)
plt.legend()
plt.show()
