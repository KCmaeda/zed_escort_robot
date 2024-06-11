#!/usr/bin/env python3

########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################


#   This sample shows how to detect a human bodies and draw their 
#   modelised skeleton in an OpenGL window

### zed_camera library ###
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import ogl_viewer.tracking_viewer as glpos
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import argparse
import json

### human_path predicition library ###
import math
import matplotlib.pyplot as plt
from IPython.display import display
import time
from scipy.spatial.transform import Rotation
import copy
from numpy.random import default_rng
from lmfit.models import SineModel
from lmfit.models import ExpressionModel
from lmfit import Parameters

### ROS library ###
import rospy
import math
import tf
import geometry_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionResult
import time
import quaternion

from matplotlib.animation import FuncAnimation


#variables possible to change frequently
plot_number = 50
#interval of fitting sine curve to the torsion angle
fitting_interval = 1 #[s]

#standard deviation of kalman filter for the angular velocity of trajectry(these were adjusted. I don't want you to change them)
#standard deviation of angle
std_theta = 1
#standard deviation of angular velocity
std_omega = 2
#standard deviation of angle observation
std_theta_obs = 30.0

#velocity threshold to detect whether the target is walking or stops
velocity_threshold = 0.2

#value used for prediction
#tilt of the regression line, it was decided by my feeling, you can change this.
alpha = 10
#intercept of the regression line, it was decided by my feeling, you can change this.
beta = 5
#coefficient of the angular velocity predicted by the torsion angle in prediction fomuila. it is composed of a variance(accuracy)
A_p = 20/(20+30)
#coefficient of the angular velocity predicted by the conventional method in prediction fomuila. it is composed of a variance(accuracy)
A_conv = 30/(20+30)
###------------------------------------------------------------


### adjust the body_data(position,velocity) ###
def addIntoOutput(out, identifier, tab):
    out[identifier] = {}
    for i in range(len(tab)):
        out[identifier][i] = str(tab[i])
    return out

### adjust the data of body_data(keypoint:x,y,z,confidence) ###
def addIntoOutputKey(out, identifier, tab):
    out[identifier] = {}
    for i in range(len(tab)):
        out[identifier][i] = {}
        out[identifier][i]['x'] = str(tab[i][0])
        out[identifier][i]['y'] = str(tab[i][1])
        out[identifier][i]['z'] = str(tab[i][2])
    return out

### store the body_data ###
def serializeBodyData(body_data):
    #Serialize BodyData into a JSON like structure
    out = {}
    out["id"] = body_data.id
    out["unique_object_id"] = str(body_data.unique_object_id)
    out["tracking_state"] = str(body_data.tracking_state)
    out["action_state"] = str(body_data.action_state)
    addIntoOutput(out, "position", body_data.position)
    addIntoOutput(out, "velocity", body_data.velocity)
    addIntoOutputKey(out, "keypoint", body_data.keypoint)
    addIntoOutput(out, "keypoint_confidence", body_data.keypoint_confidence)
    return out

def serializeBodies(bodies):
    #Serialize Bodies objects into a JSON like structure#
    out = {}
    out["is_new"] = bodies.is_new
    out["is_tracked"] = bodies.is_tracked
    out["timestamp"] = bodies.timestamp.data_ns
    out["body_list"] = {}
    c_body = 0
    for sk in bodies.body_list:
        out["body_list"][c_body] = serializeBodyData(sk)
        c_body += 1
    return out

#to save json file containing np.array
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

### save the body_data in json file ###
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

### set parameters of zed_camera ###
def parse_args(init):
    if len(opt.input_svo_file)>0 and opt.input_svo_file.endswith(".svo"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address)>0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("HD2K" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")

#to calculate torsion angle that unit is degree
def torsionAngle(joint):
    #extract joint position
    #right hip
    RH_d = joint[22]
    #left hip
    LH_d = joint[18]
    #right shoulder
    RS_d = joint[11]
    #left shoulder
    LS_d = joint[4]
    
    Rhip = np.array(RH_d, dtype="float32")
    Lhip = np.array(LH_d, dtype="float32")
    Rshoulder = np.array(RS_d, dtype="float32")
    Lshoulder = np.array(LS_d, dtype="float32")
    
    #hip vector
    dif_hip = Lhip - Rhip
    #shoulder vector
    dif_shoulder = Lshoulder - Rshoulder
    #to calculate torsion angle in xz plane because y-axis equals height. 
    torsion_angle_frame = 180 / math.pi * ( math.atan2(dif_shoulder[2], dif_shoulder[0]) - math.atan2(dif_hip[2], dif_hip[0]) )
    
    return torsion_angle_frame #[deg]


#to extract the target position, velocity and speed
#the return array is [x, y, z, vx, vy, vz, theta], xz is horion. y is height
def targetPose(joint):
    out = np.full(7, np.nan, dtype="float32")
    #x
    out[0] = joint.position[0] #[m]
    #y
    out[1] = joint.position[1] #[m]
    #z
    out[2] = joint.position[2] #[m]
    #element x of velocity vector
    out[3] = joint.velocity[0] #[m/s]
    #element y
    out[4] = joint.velocity[1] #[m/s]
    #element z
    out[5] = joint.velocity[2] #[m/s]
    #speed of xz horizontal plane
    out[6] = math.sqrt(joint.velocity[0]**2 + joint.velocity[2]**2) #[m/s]

    return out


#declaration of dict to save data
#input is length of initial numpy.array. it's at least 1
def makeDictTarget(len_array):
    #dict to return
    out = {}
    
    #if the length is 1, array is not made in row direction
    if len_array == 1:
        out["time"] = np.nan
        out["count"] = 0
        out["torsion_angle"] = np.nan
        out["pose_obs"] = np.full(7, np.nan, dtype="float32")
        out["angle_obs"] = np.nan
        out["angvel_obs"] = np.nan
        out["angle_kf_obs"] = np.nan
        out["angvel_kf_obs"] = 0
        out["P_kf_obs"] = np.array([10, 0, 0, 30], dtype="float32")
        #target
        out["pose"] = np.full(7, np.nan, dtype="float32")
        out["angle"] = np.nan
        out["angvel"] = np.nan
        out["angle_kf"] = np.nan
        out["angvel_kf"] = 0
        out["P_kf"] = np.array([10, 0, 0, 30], dtype="float32")
        out["sinfit"] = np.zeros(len_array, dtype="float32")
        out["average"] = np.nan

        #prediction
        out["target_x_predicted"] = np.nan
        out["target_z_predicted"] = np.nan
        out["target_angle_predicted"] = np.nan
        #prediction
        out["target_x_predicted_conv"] = np.nan
        out["target_z_predicted_conv"] = np.nan
        out["target_angle_predicted_conv"] = np.nan

    else:
        #target from zed camera, that is in relative coordinate
        out["time"] = np.full(len_array, np.nan, dtype="float32")
        out["count"] = np.zeros(len_array, dtype="int32")
        out["torsion_angle"] = np.full(len_array, np.nan, dtype="float32")
        out["pose_obs"] = np.full((len_array, 7), np.nan, dtype="float32")
        out["angle_obs"] = np.full(len_array, np.nan, dtype="float32")
        out["angvel_obs"] = np.full(len_array, np.nan, dtype="float32")
        out["angle_kf_obs"] = np.full(len_array, np.nan, dtype="float32")
        out["angvel_kf_obs"] = np.full(len_array, np.nan, dtype="float32")
        out["P_kf_obs"] = np.full((len_array, 4), np.nan, dtype="float32")
         #initial value
        out["angvel_kf_obs"][-1] = 0
        out["P_kf_obs"][-1] = np.array([10, 0, 0, 30], dtype="float32")
        #target in absolute coordinate
        out["pose"] = np.full((len_array, 7), np.nan, dtype="float32")
        out["angle"] = np.full(len_array, np.nan, dtype="float32")
        out["angvel"] = np.full(len_array, np.nan, dtype="float32")
        out["angle_kf"] = np.full(len_array, np.nan, dtype="float32")
        out["angvel_kf"] = np.full(len_array, np.nan, dtype="float32")
        out["P_kf"] = np.full((len_array, 4), np.nan, dtype="float32")
         #initial value
        out["angvel_kf"][-1] = 0
        out["P_kf"][-1] = np.array([10, 0, 0, 30], dtype="float32")
        #for sine fitting. actually sinfit is unnecessary
        out["sinfit"] = np.zeros((len_array, 4), dtype="float32")
        out["average"] = np.full(len_array, np.nan)
        #prediction
        out["target_x_predicted"] = np.full(len_array, np.nan, dtype="float32")
        out["target_z_predicted"] = np.full(len_array, np.nan, dtype="float32")
        out["target_angle_predicted"] = np.full(len_array, np.nan, dtype="float32")
        #conventional
        out["target_x_predicted_conv"] = np.full(len_array, np.nan, dtype="float32")
        out["target_z_predicted_conv"] = np.full(len_array, np.nan, dtype="float32")
        out["target_angle_predicted_conv"] = np.full(len_array, np.nan, dtype="float32")

    return out

#to store data to dict to save
def dictContain(dict_out, key, input): 
    dict_out[key] = np.concatenate([dict_out[key], [input]], axis = 0)

    
#to fit sine curve to the torsion angle. unnecessary
def sinFitting(frame, torsion, delta_t, pre_param):
    
    #declaration of list to delete nan
    list_delete = []
    #repetition at each frame
    for i_s in range(0, len(fitX)):
        #to detect nan. both of values are deleted even if the either one is nan.
        if np.isnan(fitX[i_s]) or np.isnan(fitY[i_s]):
            #to contain the index
            list_delete.append(i_s)

    #to delete nan
    fitX = np.delete(fitX, list_delete, 0)
    fitY = np.delete(fitY, list_delete, 0)
    
    #formula of sine curce to fit the torsion angle
    model = ExpressionModel("amp * sin(freq * (x - initial_phase)) + b")
    #declaration of parameter
    params = Parameters()
    #initial definition of parameters
    params.add(name = "amp", value = pre_param[0], min= 0.01, max = 90.0)
    params.add(name = "freq", value = pre_param[1])
    params.add(name = "initial_phase", value = pre_param[2])
    params.add(name = "b", value = pre_param[3])
    #sine fitting
    result = model.fit(fitY, params, x=fitX)
    #to extract estimated parameters
    params_result = result.params.valuesdict()

    return params_result
#to calculate the average of torsion angle in the fitting interval
def intervalAverage(time, torsion):
    #index to determine the length of torsion angle to fit sine curve
    for i in range(1, len(time)+1):
        if time[-i] <= time[-1] - fitting_interval:
            index_interval = -i-1
    #extract torsion angle by the length
    fitY = torsion[index_interval:]
    return np.nanmean(fitY)

#to shift the trajectory angle mistakend by a problem about argument range of arctan2
#1st argument, angle is predicted by kalman filter
#2nd argument, angle_obs is observed by ZED camera
def shift_angle(angle, angle_obs):
    #absolute value of difference between 2 arguments
    abs_dif = abs(angle - angle_obs)
    #sign of the difference
    sign = lambda x: math.copysign(1, x)
    #to detect how the trajectory angle is mistaken
    for i in range(0, 100):
        if i == 0:
            #when the difference is small
            if (abs_dif < 360*i + 3*std_theta_obs):
                return angle_obs
            #when the difference is great and it is close to 180
            elif (abs_dif >= 360*i + 3*std_theta_obs) and (abs_dif < 360*(i+1) - 3*std_theta_obs):
                return angle_obs + 180*(2*i+1) * sign(angle - angle_obs)
        else:
            #when the difference is close to 360*i
            if (abs_dif < 360*i + 3*std_theta_obs):
                return angle_obs + 360*i * sign(angle - angle_obs)
            #when the difference is close to 180*(2*i+1)
            elif (abs_dif >= 360*i + 3*std_theta_obs) and (abs_dif < 360*(i+1) - 3*std_theta_obs):
                return angle_obs + 180*(2*i+1) * sign(angle - angle_obs)
            

#kalman filter to calculate angular velocity for real time
#1st, 2nd and 4th arguments is the values of before step
#3rd argument is observed angle, 5th is sampling interval
def kf_angvel(angle, angvel, angle_observed, P_angvel, delta_t):
    #covarriance of process update
    U = np.array([[std_theta**2, 0], [0, std_omega**2]], dtype="float32")
    #covarriance of observed update
    Z = std_theta_obs ** 2
    #reshape error covariance
    P_angvel = np.array([[P_angvel[0], P_angvel[1]], [P_angvel[2], P_angvel[3]]], dtype="float32")

    #time update-------------------------------------------------------------------
    #coefficient of the target state
    A = np.array([[1, delta_t], [0, 1]], dtype="float32")
    #coefficient of the process noise 
    B = np.array([[1, 0], [0, 1]], dtype="float32")
    #Pre estimated error covariance
    P_bar = A @ P_angvel @ A.T + B @ U @ B.T
    #Pre-estimated value
    x_bar = A @ np.array([angle, angvel], dtype="float32")

    #observe update---------------------------------------------------------------
    #if the observed angle is nan, observed update is not implemented
    if np.isnan(angle_observed):
        #to reshape the error covariance
        P_export = np.array([P_bar[0, 0], P_bar[0, 1], P_bar[1, 0], P_bar[1, 1]], dtype="float32")
        #print("wowowbar", x_bar[0], x_bar[1], P_export)
        return x_bar[0], x_bar[1], P_export
    
    else:
        #coefficient of observation
        C = np.array([1, 0], dtype="float32")
        #value by observation model and pre-estimated value
        y_bar = C @ x_bar
        #to shift the trajectory angle mistakend by a problem about argument range of arctan2
        angle_observed = shift_angle(y_bar, angle_observed)
        #kalman gain
        KG = P_bar @ C / ( C.T @ P_bar @ C + Z)
        #post-estimated value
        X_est = x_bar + KG * (angle_observed - y_bar)
        #post-estimated error covariance
        P_est = (np.identity(len(P_bar), dtype="float32") - KG @ C) @ P_bar
        #to reshape the error covariance
        P_export = np.array([P_est[0, 0], P_est[0, 1], P_est[1, 0], P_est[1, 1]], dtype="float32")
        
        return X_est[0], X_est[1], P_export

#to store temporary dict to other dict to write json.file
def dictContainFor(dict_target, dict_target_temp, frame):
    #repetition at each targets
    for target_id in dict_target.keys():
        #if the dict with target_id is not in dict_target_temp
        if not target_id in dict_target_temp.keys():
            dict_target_temp[target_id] = {}
        #if the velocity is under the threshold, the values from kalman filter are made nan or initial value.
        if "pose" in dict_target_temp[target_id].keys():
            if dict_target_temp[target_id]["pose"][6] < velocity_threshold:
                dict_target_temp[target_id]["angle_kf_obs"] = np.nan
                dict_target_temp[target_id]["angvel_kf_obs"] = np.nan
                dict_target_temp[target_id]["P_kf_obs"] = np.array([10, 0, 0, 30], dtype="float32")
                dict_target_temp[target_id]["angle_kf"] = np.nan
                dict_target_temp[target_id]["angvel_kf"] = np.nan
                dict_target_temp[target_id]["P_kf"] = np.array([10, 0, 0, 30], dtype="float32")
        #repetition at each keys
        for key in dict_target[target_id].keys():
            #if the key is not in dict_target_temp[target_id], nan or zero is assigned
            if not key in dict_target_temp[target_id].keys():
                if key == "sinfit":
                    dict_target_temp[target_id][key] = np.zeros(4, dtype="float32")
                elif key == "pose" or key == "pose_obs":
                    dict_target_temp[target_id][key] = np.full(7, np.nan, dtype="float32")
                elif key == "time":
                    dict_target_temp[target_id][key] = frame[-1]
                elif key == "count":
                    dict_target_temp[target_id][key] = 0
                elif key == "P_kf_obs" or key == "P_kf":
                    dict_target_temp[target_id][key] = dict_target[target_id][key][-1, :]
                else:
                    dict_target_temp[target_id][key] = np.nan

            #to store data to dict to save
            dictContain(dict_target[target_id], key, dict_target_temp[target_id][key])


# Variable definition for robot position
x_robot = 0
y_robot = 0
th_robot = 0
dx_robot = 0
dy_robot = 0
dth_robot = 0
current_time_robot = 0

# A function to convert from the robot's pose (quaternion) to the robot's z angle.
def quaternion_to_euler_angle(w, x, y, z):
    # Convert quaternion to euler angle around z-axis
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return yaw

# A function that receives the robot's current position and assigns it to a variable.
def odom_callback(data):
    global x_robot
    global y_robot
    global th_robot
    global dx_robot
    global dy_robot
    global dth_robot
    global current_time_robot
    x_robot = data.pose.pose.position.x
    y_robot = data.pose.pose.position.y
    th_robot = quaternion_to_euler_angle(data.pose.pose.orientation.w, 0.0, 0.0, data.pose.pose.orientation.z)
    dx_robot = data.twist.twist.linear.x
    dy_robot = data.twist.twist.linear.y
    dth_robot = data.twist.twist.angular.z
    current_time_robot = data.header.stamp.to_sec()

# Setting diagrams showing robot positions and human positions
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlim(-8, 8)
ax.set_ylim(-8, 8)
arrow_length = 0.2
robot_plot, = ax.plot([], [], 'bo', label='Robot')
robot_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='b', ec='b')
human_plot, = ax.plot([], [], 'ro', label='Human')
human_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='r', ec='r')
goal_plot, = ax.plot([], [], 'go', label='Goal')
goal_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='g', ec='g')
human_f_plot, = ax.plot([], [], 'ko', label='Human_f')
human_f_arrow = ax.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='k', ec='k')
ax.legend()
ax.set_xlabel('X position')
ax.set_ylabel('Y position')

# A function for creating animations showing the positions of robots and people
def update_plot(x_robot, y_robot, th_robot, x_human, y_human, th_human, goal_x, goal_y, pub_th, x_human_f, y_human_f, th_human_f):
    global robot_arrow, human_arrow, goal_arrow, human_f_arrow
    if robot_arrow:
        robot_arrow.remove()
    if human_arrow:
        human_arrow.remove()
    if goal_arrow:
        goal_arrow.remove()
    if human_f_arrow:
        human_f_arrow.remove()
    robot_plot.set_data(x_robot, y_robot)
    robot_arrow = ax.arrow(x_robot, y_robot, arrow_length * np.cos(th_robot), arrow_length * np.sin(th_robot), head_width=0.2, head_length=0.2, fc='k', ec='k')
    human_plot.set_data(x_human, y_human)
    human_arrow = ax.arrow(x_human, y_human, arrow_length * np.cos(th_human), arrow_length * np.sin(th_human), head_width=0.2, head_length=0.2, fc='k', ec='k')
    goal_plot.set_data(goal_x, goal_y)
    goal_arrow = ax.arrow(goal_x, goal_y, arrow_length * np.cos(pub_th), arrow_length * np.sin(pub_th), head_width=0.2, head_length=0.2, fc='k', ec='k')
    human_f_plot.set_data(x_human_f, y_human_f)
    human_f_arrow = ax.arrow(x_human_f, y_human_f, arrow_length * np.cos(th_human_f), arrow_length * np.sin(th_human_f), head_width=0.2, head_length=0.2, fc='k', ec='k')
    plt.draw()
    plt.pause(0.001)


def main():
    print("Running Body Tracking sample ... Press 'q' to quit, or 'm' to pause or restart")

    global x_robot
    global y_robot
    global th_robot
    global dx_robot
    global dy_robot
    global dth_robot
    global current_time_robot
    
    ## ros publish initial setting ##
    rospy.init_node('bodyTracking', anonymous=True)
    # goal_pub = rospy.Publisher('/goal', PoseStamped, queue_size=10)
    # odom_subscriber = rospy.Subscriber('/hsrb/odom', Odometry, odom_callback)
    goal_pub = rospy.Publisher('move_base_simple/goal', PoseStamped, queue_size=10) # robot's goal
    odom_subscriber = rospy.Subscriber('/odom', Odometry, odom_callback) # robot's position
    odom_pub_human = rospy.Publisher("odom_human_abs", Odometry, queue_size=50) # human's position (realtime)
    odom_pub_human_f = rospy.Publisher("odom_human_abs_f", Odometry, queue_size=50) # human's position (future)
    odom_human_broadcaster = tf.TransformBroadcaster()
    odom_human_f_broadcaster = tf.TransformBroadcaster()
    current_time_robot = rospy.Time.now()
    current_time_human = rospy.Time.now()
    time.sleep(0.1)
    last_time_robot= rospy.Time.now()
    last_time_human= rospy.Time.now()
    dt_robot = (current_time_robot - last_time_robot).to_sec()
    dt_human = (current_time_human - last_time_human).to_sec()
    publish_rate = 1.0 # publish interval is about the value[seconds] of publish_rate 
    count_pub = 0.0
    pre_goal_x = 0.0
    pre_goal_y = 0.0
    pre_goal_th = 0.0
    pre_th_human = 0.
    pre_x_human = 0.
    pre_y_human = 0.
    th_TorF = 0
    x_TorF = 0
    y_TorF = 0


    ## zed_camera and zed_viewer setting ##
    # Create a InitParameters object and set configuration parameters #
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.VGA  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    parse_args(init_params)
    # Create a Camera object
    zed = sl.Camera()
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    runtime = sl.RuntimeParameters()
    camera_pose = sl.Pose()
    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    positional_tracking_parameters.enable_imu_fusion = True 
    # If the camera is static, uncomment the following line to have better performances
    positional_tracking_parameters.set_as_static = False #True
    err = zed.enable_positional_tracking(positional_tracking_parameters)
    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking = True                # Track people across images flow
    body_param.enable_body_fitting = True #False            # Smooth skeleton move
    body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST  #ACCURATE
    body_param.body_format = sl.BODY_FORMAT.BODY_34  # Choose the BODY_FORMAT you wish to use
    # Enable Object Detection module
    zed.enable_body_tracking(body_param)
    body_runtime_param = sl.BodyTrackingRuntimeParameters()
    body_runtime_param.detection_confidence_threshold = 40
    # Get ZED camera information
    camera_info = zed.get_camera_information()
    # Create OpenGL viewer
    viewer_position = glpos.GLViewer()
    viewer_position.init(camera_info.camera_model)
    if opt.imu_only:
        sensors_data = sl.SensorsData()
    py_translation = sl.Translation()
    pose_data = sl.Transform()
    text_translation = ""
    text_rotation = ""
    file = open('output_trajectory.csv', 'w')
    file.write('tx, ty, tz \n')
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1280), min(camera_info.camera_configuration.resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_configuration.resolution.width
                 , display_resolution.height / camera_info.camera_configuration.resolution.height]
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_configuration.calibration_parameters.left_cam, body_param.enable_tracking,body_param.body_format)
    # Create ZED objects filled in the main loop
    bodies = sl.Bodies()
    image = sl.Mat()
    key_wait = 10 
    skeleton_file_data = {}

    #declaration of dict to save jsonfile
    dict_target = {}
    dict_target_pre = {}
    dict_target_temp = {}
    #np.array to calculate average of the torsion angle
    array_torsion_save = np.full(60, np.nan)
    #array to store rotation matrix of ZED
    ZED_rotation = np.full((plot_number, 3), np.nan, dtype="float32")
    #array to store position, velocity vector, speed of ZED
    ZED_pose = np.full((plot_number, 7), np.nan, dtype="float32")
    #initial value 
    ZED_pose[-1, :3] = np.array([0, 0, 0], dtype="float32")
    #ro count frane since tha capture is started
    countframe = 0
    #it is frame but actually it is time
    frame = np.array([0], dtype="float32")

    while viewer.is_available() and viewer_position.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            #time-------------------
            if countframe == 0:
                timestamp = time.perf_counter()
                delta_t = 0.1
            else:
                delta_t = (time.perf_counter() - timestamp)
                timestamp = time.perf_counter()
            #framz
            frame = np.concatenate([frame, [frame[-1] + delta_t]])
            #frame[-1] = countframe
            countframe += 1
            #----------------------------

            tracking_state = zed.get_position(camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
            if opt.imu_only :
                if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                    rotation = sensors_data.get_imu_data().get_pose().get_euler_angles()
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                    viewer_position.updateData(sensors_data.get_imu_data().get_pose(), text_translation, text_rotation, tracking_state)
            else : 
                if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                    #Get rotation and translation and displays it
                    rotation = camera_pose.get_rotation_vector()
                    translation = camera_pose.get_translation(py_translation)
                    ZED_rotation = np.roll(ZED_rotation, -1, axis=0)
                    ZED_rotation[-1, :] = rotation
                    ZED_velocity = (translation.get()[:] - ZED_pose[-1, :3]) / delta_t
                    ZED_pose_add = np.concatenate([np.concatenate([translation.get()[:], ZED_velocity]), [np.linalg.norm([ZED_velocity[0], ZED_velocity[2]])]])
                    ZED_pose = np.roll(ZED_pose, -1, axis=0)
                    ZED_pose[-1, :] = ZED_pose_add


                    ZED_pose[-1, 2] = x_robot #(x_robot)* math.cos(math.pi) - (y_robot)* math.sin(math.pi) 
                    translation.get()[2] = x_robot
                    ZED_pose[-1, 0] = y_robot #(x_robot)* math.sin(math.pi) + (y_robot)* math.cos(math.pi)
                    translation.get()[0] = y_robot
                    ZED_pose[-1, 5] = dx_robot
                    ZED_pose[-1, 3] = dy_robot
                    #print(ZED_rotation[-1, 1])
                    ZED_rotation[-1, 1] =   th_robot
                    rotation[1] =   th_robot
                    ZED_velocity[2] = dx_robot
                    ZED_velocity[0] = dy_robot

                    text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))
                    pose_data = camera_pose.pose_data(sl.Transform())
                    #print("rotation" + str(rotation))
                    text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                    file.write(str(translation.get()[0])+", "+str(translation.get()[1])+", "+str(translation.get()[2])+"\n")

                    ##  count publishing rate##
                    count_pub += delta_t
                    if count_pub >= publish_rate:
                        last_time_robot = rospy.Time.now()
                        count_pub = 0

                viewer_position.updateData(pose_data, text_translation, text_rotation, tracking_state)
            
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve bodies
            zed.retrieve_bodies(bodies, body_runtime_param)
            #skeleton_file_data[str(bodies.timestamp.get_milliseconds())] = serializeBodies(bodies)
            


            ## escorting human ##
            ##############################escorting###################################################
            #copy of dict at before step
            if len(dict_target_temp) > 1:
                dict_target_pre = copy.deepcopy(dict_target_temp)
            #declaration of dict to record data of present dict
            dict_target_temp = {}
            #repetition at each targets who are captured by ZED camera
            #However I assume that the number of targets is only 1.
            for sk in bodies.body_list:
                #definition of target id. it's like 0, 1, 2,...
                target_id = str(sk.id)
                #declaration of dict to save data when at the first time
                if not target_id in dict_target.keys():
                    #to export data
                    dict_target[target_id] = makeDictTarget(50)
                    #instead of before step dict
                    dict_target_pre[target_id] = makeDictTarget(1)
                
                #declaration of dict of target_id
                dict_target_temp[target_id] = {}
                #to count frame
                dict_target_temp[target_id]["count"] = dict_target_pre[target_id]["count"] + 1

                #target oberved--------------------------------------------------------------------------------------------------
                #torsion angle
                torsion_angle = torsionAngle(sk.keypoint) #[deg]
                dict_target_temp[target_id]["torsion_angle"] = torsion_angle
                #to save the angle in order to calculate average of torsion angle instead of offset of sine fitting
                array_torsion_save = np.roll(array_torsion_save, -1)
                array_torsion_save[-1] = torsion_angle
                #target pose[m],velocity[m/s], speed[m/s]
                #the return array is [x, y, z, vx, vy, vz, theta], xz is horion. y is height
                dict_target_temp[target_id]["pose_obs"] = targetPose(sk)
                #target angle(walking direction)[deg]     atan2(human_ZEDz_velocity, human_ZEDx_velocity)
                target_angle_out = math.atan2(dict_target_temp[target_id]["pose_obs"][5], dict_target_temp[target_id]["pose_obs"][3]) * 180 / math.pi

                dict_target_temp[target_id]["angle_obs"] = target_angle_out
                #yaw angle of ZED camera
                #I don't assume that the other angle of ZED camera are change. They must be 0.
                angle_ZED_y = ZED_rotation[-1, 1]
                #rotation matrix[rad] to convert the camera coordinate to the world coordinate
                R_inv = np.array([[math.cos(angle_ZED_y), math.sin(angle_ZED_y)], [-math.sin(angle_ZED_y), math.cos(angle_ZED_y)]], dtype="float32")
                ##to convert the camera coordinate to the world coordinate
                #target pose
                target_pose_out = np.zeros(7, dtype="float32")
                #to convert xz plane for the yaw angle
                target_xz = R_inv @ np.array([dict_target_temp[target_id]["pose_obs"][0], dict_target_temp[target_id]["pose_obs"][2]]) + np.array([ZED_pose[-1, 0], ZED_pose[-1, 2]], dtype="float32")
                target_pose_out[0] = target_xz[0]
                target_pose_out[2] = target_xz[1]
                #y axis(height) is just added by the height of ZED camera.
                target_pose_out[1] = dict_target_temp[target_id]["pose_obs"][1] + ZED_pose[-1, 1]
                #to convert xz velocity for the yaw angle
                target_xz_velocity = R_inv @ np.array([dict_target_temp[target_id]["pose_obs"][3] + ZED_pose[-1, 3], dict_target_temp[target_id]["pose_obs"][5] + ZED_pose[-1, 5]], dtype="float32")
                target_pose_out[3] = target_xz_velocity[0]
                target_pose_out[5] = target_xz_velocity[1]
                #y axis(height direction of velocity) is just added by the height of ZED camera.
                target_pose_out[4] = dict_target_temp[target_id]["pose_obs"][4] + ZED_pose[-1, 4]
                #target speed
                target_pose_out[6] = np.linalg.norm(np.array([target_pose_out[3], target_pose_out[5]]))
                
                dict_target_temp[target_id]["pose"] = target_pose_out
                #target angle[deg]
                dict_target_temp[target_id]["angle"] = dict_target_temp[target_id]["angle_obs"] + angle_ZED_y * 180 / math.pi
                #to retreive nan from target angle not to disturbe kalman filter to calculate angular velocity
                #and to determine whether walking or stopping by the walking speed
                if np.isnan(dict_target_pre[target_id]["angle_kf"]) and dict_target_temp[target_id]["pose"][6] > velocity_threshold:
                    angle_obs_initial = target_angle_out
                    angvel_obs_initial = 0.0
                #if the speed is under the threshold, I make the angular velocity 0.
                elif np.isnan(dict_target_pre[target_id]["angle_kf"]) and not(dict_target_temp[target_id]["pose"][6] > velocity_threshold):
                    angle_obs_initial = 0.0
                    angvel_obs_initial = 0.0
                else:
                    angle_obs_initial = dict_target_pre[target_id]["angle_kf"]
                    angvel_obs_initial = dict_target_pre[target_id]["angvel_kf"]
                #target angular velocity just calculated by differential method. this value contains terrible noise
                target_angvel_out = (dict_target_temp[target_id]["angle"] - dict_target_pre[target_id]["angle"]) / delta_t
                dict_target_temp[target_id]["angvel"] = target_angvel_out
                #target angular velocity by kalman filter
                dict_target_temp[target_id]["angle_kf"], dict_target_temp[target_id]["angvel_kf"], dict_target_temp[target_id]["P_kf"] = kf_angvel(
                    angle_obs_initial, angvel_obs_initial, 
                    dict_target_temp[target_id]["angle"], dict_target_pre[target_id]["P_kf"][:], delta_t)
                #to calculate average or offset of the torsion angle at the fitting_interval
                if dict_target_temp[target_id]["count"] > plot_number:
                    #offset of sine fitting. It needs heavy computation and sampling freq becomes low
                    #dict_s = sinFitting(frame, dict_target[target_id]["torsion_angle"], delta_t, dict_target[target_id]["sinfit"][-1, :])
                    #param_sinfit = np.array([dict_s["amp"], dict_s["freq"], dict_s["initial_phase"], dict_s["b"]])
                    
                    #average of the torsion angle at the fitting_interval
                    dict_target_temp[target_id]["average"] = intervalAverage(frame, array_torsion_save)                
                
                #prediction with the torsion angle
                angvel_present = dict_target_temp[target_id]["angvel_kf"]  /180*math.pi
                #if the average can be calculated,
                if "average" in dict_target_temp[target_id].keys():
                    #walking angular velocity is predicted by regression line
                    #the unit is rad/s
                    angvel_est = (alpha * dict_target_temp[target_id]["average"] + beta) /180*math.pi #[rad/s]
                else:
                    #if the average cannot be calculated, the present angular_velocity is used.
                    #the unit is rad/s
                    angvel_est = dict_target_temp[target_id]["angvel_kf"]  /180*math.pi #[rad/s]
                
                #the angular_velocity predicted with conventional method is to be present one
                angvel_conv = angvel_present
                #present target angle, the unit is rad
                angle_present = dict_target_temp[target_id]["angle"] /180*math.pi #[rad]
                #target angle predicted for 1 second future
                angle_est_1s = angle_present + 1/2*(A_p * angvel_est + A_conv * angvel_conv)
                #proposed
                #coefficient
                A_for_x = dict_target_temp[target_id]["pose"][6] / angvel_present * ( math.sin(angle_present + angvel_present * delta_t) - math.sin(angle_present) )
                B_for_x = dict_target_temp[target_id]["pose"][6] / angvel_est * ( math.sin(angle_est_1s + angvel_est * delta_t) - math.sin(angle_est_1s) )
                C_for_y = dict_target_temp[target_id]["pose"][6] / angvel_present * ( -math.cos(angle_present + angvel_present * delta_t) + math.cos(angle_present) )
                D_for_y = dict_target_temp[target_id]["pose"][6] / angvel_est * ( -math.cos(angle_est_1s + angvel_est * delta_t) + math.cos(angle_est_1s) )
                #target position predicted for 1 second future
                target_est_x_1s = dict_target_temp[target_id]["pose"][0] + (A_for_x + B_for_x) * 1 / delta_t
                target_est_z_1s = dict_target_temp[target_id]["pose"][2] + (C_for_y + D_for_y) * 1 / delta_t
                #to store predicted ones with proposed method to dict
                dict_target_temp[target_id]["target_x_predicted"] = target_est_x_1s
                dict_target_temp[target_id]["target_z_predicted"] = target_est_z_1s
                dict_target_temp[target_id]["target_angle_predicted"] = angle_est_1s

                #conventional
                #target angle predicted for 1 second future
                angle_conv_1s = angle_present + angvel_conv
                #coefficient
                A_for_x = dict_target_temp[target_id]["pose"][6] / angvel_present * ( math.sin(angle_present + angvel_present * delta_t) - math.sin(angle_present) )
                B_for_x = dict_target_temp[target_id]["pose"][6] / angvel_conv * ( math.sin(angle_conv_1s + angvel_conv * delta_t) - math.sin(angle_conv_1s) )
                C_for_y = dict_target_temp[target_id]["pose"][6] / angvel_present * ( -math.cos(angle_present + angvel_present * delta_t) + math.cos(angle_present) )
                D_for_y = dict_target_temp[target_id]["pose"][6] / angvel_conv * ( -math.cos(angle_conv_1s + angvel_conv * delta_t) + math.cos(angle_conv_1s) )
                #target position predicted for 1 second future
                target_conv_x_1s = dict_target_temp[target_id]["pose"][0] + (A_for_x + B_for_x) * 1 / delta_t
                target_conv_z_1s = dict_target_temp[target_id]["pose"][2] + (C_for_y + D_for_y) * 1 / delta_t
                #to store predicted ones with conventional method to dict
                dict_target_temp[target_id]["target_x_predicted_conv"] = target_conv_x_1s
                dict_target_temp[target_id]["target_z_predicted_conv"] = target_conv_z_1s
                dict_target_temp[target_id]["target_angle_predicted_conv"] = angle_conv_1s


                ## ros publish human_position ##
                if count_pub == 0:
                    # calculate human_position (realtime) #
                    current_time_human = rospy.Time.now()
                    dt_human = (current_time_human - last_time_human).to_sec() # if we need velocity, we should use difference of time.
                    last_time_human = rospy.Time.now()
                    human_absxy = np.array([x_robot, y_robot]) + np.dot(np.array([[math.cos(th_robot), -math.sin(th_robot)], [math.sin(th_robot), math.cos(th_robot)]]), np.array([sk.keypoint[11][2], sk.keypoint[11][0]]))
                    x_human = human_absxy[0]
                    y_human = human_absxy[1]
                    th_human = (th_robot) + math.atan2(sk.keypoint[11][0]-sk.keypoint[4][0], sk.keypoint[11][2]-sk.keypoint[4][2]) + math.pi/2
                    
                    # publish human_position (realtime) #
                    odom_quat_human = tf.transformations.quaternion_from_euler(0, 0, th_human)
                    odom_human_broadcaster.sendTransform((x_human, y_human, 0.), odom_quat_human, current_time_human, "base_link_human", "odom_human")
                    odom_human = Odometry()
                    odom_human.header.stamp = current_time_human
                    odom_human.header.frame_id = "odom_human"
                    odom_human.child_frame_id = "base_link_human"
                    odom_human.pose.pose = Pose(Point(x_human, y_human, 0.), Quaternion(*odom_quat_human))
                    odom_human.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0)) # if we need velocity, we should change this line.
                    odom_pub_human.publish(odom_human)
                    
                    # calculate human_position (future) #
                    current_time_human_f = rospy.Time.now()
                    #human_absxy_f = np.array([x_robot, y_robot]) + np.dot(np.array([[math.cos(th_robot*2), -math.sin(th_robot*2)], [math.sin(th_robot*2), math.cos(th_robot*2)]]), np.array([dict_target[target_id]["target_z_predicted"][-20:][-1], dict_target[target_id]["target_x_predicted"][-20:][-1]]))
                    #x_human_f = human_absxy_f[0]
                    #y_human_f = human_absxy_f[1]
                    x_human_f = dict_target[target_id]["target_z_predicted"][-20:][-1] #target_pose_out[2]
                    y_human_f = dict_target[target_id]["target_x_predicted"][-20:][-1] #target_pose_out[0]
                    #x_human_f = (dict_target[target_id]["target_z_predicted"][-20:][-1] - x_robot )*math.cos(th_robot*2) - (dict_target[target_id]["target_x_predicted"][-20:][-1] - y_robot)*math.sin(th_robot*2) + x_robot
                    #y_human_f = (dict_target[target_id]["target_z_predicted"][-20:][-1] - x_robot )*math.sin(th_robot*2) + (dict_target[target_id]["target_x_predicted"][-20:][-1] - y_robot)*math.cos(th_robot*2) + y_robot
                    #x_human_f = (x_human_f - x_human )*math.cos(th_robot*2) - (y_human_f - y_human)*math.sin(th_robot*2) + x_human
                    #y_human_f = (x_human_f - x_human )*math.sin(th_robot*2) + (y_human_f - y_human)*math.cos(th_robot*2) + y_human
                    rel_x = x_human_f - x_human
                    rel_y = y_human_f - y_human
                    rotation_matrix = np.array([[np.cos(th_robot *2), -np.sin(th_robot *2)],[np.sin(th_robot *2), np.cos(th_robot *2)]])
                    rel_pos = np.array([rel_x, rel_y])
                    rotated_rel_pos = rotation_matrix @ rel_pos
                    x_human_f = rotated_rel_pos[0] + x_human
                    y_human_f = rotated_rel_pos[1] + y_human
                    th_human_f = -(dict_target_temp[target_id]["target_angle_predicted"]) +  math.pi/2 + th_robot*2

                    # publish human_position (future) #
                    odom_quat_human_f = tf.transformations.quaternion_from_euler(0, 0, th_human_f)
                    odom_human_f_broadcaster.sendTransform((x_human_f, y_human_f, 0.), odom_quat_human_f, current_time_human_f, "base_link_human_f", "odom_human_f")
                    odom_human_f = Odometry()
                    odom_human_f.header.stamp = current_time_human_f
                    odom_human_f.header.frame_id = "odom_human_f"
                    odom_human_f.child_frame_id = "base_link_human_f"
                    odom_human_f.pose.pose = Pose(Point(x_human_f, y_human_f, 0.), Quaternion(*odom_quat_human_f))
                    odom_human_f.twist.twist = Twist(Vector3(0, 0, 0), Vector3(0, 0, 0))
                    odom_pub_human_f.publish(odom_human_f)

                    D = 1.8 #0.5 # length from human to robot
                    threshold_th = 0.1 # angle threshold for publishing goal_pose
                    threshold_xy = 0.1 # position(x, y) threshold for publishing goal_pose

                    # if the value is lower than the threshold, publish the previously transmitted value.
                    if abs(abs(th_human) - abs(pre_th_human)) <= threshold_th:
                        pub_th = pre_th_human
                        th_TorF = 1
                    else:
                        pub_th = th_human
                        pre_th_human = th_human
                        th_TorF = 0
                    if abs(abs(x_human) - abs(pre_x_human)) <= threshold_xy:
                        pub_x = pre_x_human
                        x_TorF = 1
                    else:
                        pub_x = x_human
                        pre_x_human = x_human
                        x_TorF = 0
                    if abs(abs(y_human) - abs(pre_y_human)) <= threshold_xy:
                        pub_y = pre_y_human
                        y_TorF = 1
                    else:
                        pub_y = y_human
                        pre_y_human = y_human
                        y_TorF = 0

                    # Even if it stops to ensure that the goal topic is sent, it will still send one additional topic, so the threshold for whether to send or not is set to 2.
                    if th_TorF!=0:
                        goal_point = PoseStamped()
                        goal_point.header.seq = 0
                        goal_point.header.stamp = rospy.Time.now()
                        goal_point.header.frame_id = 'map'
                        goal_point.pose.position.x = pub_x + math.cos(pub_th - 3.14/4*0)*D
                        goal_point.pose.position.y = pub_y + math.sin(pub_th - 3.14/4*0)*D
                        goal_point.pose.orientation.z =  math.sin(pub_th/2)
                        goal_point.pose.orientation.w =  math.cos(pub_th/2)
                        goal_pub.publish(goal_point)
                        update_plot(x_robot, y_robot, th_robot, x_human, y_human, th_human, goal_point.pose.position.x, goal_point.pose.position.y, pub_th, x_human_f, y_human_f, th_human_f)
                    
                    time.sleep(0.01)


            dictContainFor(dict_target, dict_target_temp, frame)
            
            ## update zed_viewer ##
            # Update GL view
            viewer.update_view(image, bodies) 
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale, bodies.body_list, body_param.enable_tracking, body_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            key = cv2.waitKey(key_wait)
            if key == 113: # for 'q' key
                print("Exiting...")
                break
            if key == 109: # for 'm' key
                if (key_wait>0):
                    print("Pause")
                    key_wait = 0 
                else : 
                    print("Restart")
                    key_wait = 10 

    ## Save data into JSON file: ##
    #file_sk = open("target_"+ str(timestamp)+".json", 'w')
    #file_sk.write(json.dumps(dict_target, cls=NumpyEncoder, indent=4))
    #file_sk.close()

    ## stop zed_viewer ##
    viewer_position.exit()
    image.free(sl.MEM.CPU)
    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    plot_number = 50
    fitting_interval = 1 #[s]

    #angvel kf
    std_theta = 1
    std_omega = 2
    std_theta_obs = 30.0
    velocity_threshold = 0.1

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    parser.add_argument('--imu_only', action = 'store_true', help = 'Either the tracking should be done with imu data only (that will remove translation estimation)' )
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 

