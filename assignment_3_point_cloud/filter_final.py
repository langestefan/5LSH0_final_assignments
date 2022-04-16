#!/usr/bin/env python
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy
from scipy import ndimage as ndi
from skimage import feature

import rospy
import ros_numpy as ros_np
import numpy as np
from sensor_msgs.msg import PointCloud2


def make_grid(point_cloud, step_size):
    """
    Take point cloud and return x-y grid
    """
    # take x-y points
    x = point_cloud['x'].copy()
    y = point_cloud['y'].copy()
    
    # compute min-max grid edges
    x_min = -np.max(-x)
    x_max = np.max(x)
    y_min = -np.max(-y)
    y_max = np.max(y)
    
    # define grid size
    x_size = np.int_((np.ceil(x_max) - np.floor(x_min)) / step_size)
    y_size = np.int_((np.ceil(y_max) - np.floor(y_min)) / step_size)
    
    # define np array to store accumulated points
    sum_2d = np.zeros((x_size, y_size))
    
    # figure out for all our grid indices how many points fall within it
    for x_i in range(x_size):
    
        x_coord = np.floor(x_min) + step_size * x_i
        
        # loop over the y-axis
        for y_i in range(y_size):
            y_coord = np.ceil(y_min) + step_size * y_i
            
            result_x = np.where((x_coord < x) & (x <= x_coord + step_size))
            result_y = np.where((y_coord < y) & (y <= y_coord + step_size))
            
            if np.size(result_x) > np.size(result_y):
                n_points = np.isin(result_x, result_y).sum() 
            else:
                n_points = np.isin(result_y, result_x).sum()
                
            sum_2d[x_i, y_i] = n_points
            
    print('sum_2d: {0}'.format(np.shape(sum_2d)))
    return sum_2d                            


class pc_filter:
    def __init__(self):
        self.pc_sub = rospy.Subscriber('/cloud_map', PointCloud2, self.callback)
        self.pc_pub = rospy.Publisher('/filtered_pc', PointCloud2, queue_size=50)
        
        self.old_length = 0
        self.msg_counter = 0
        

    def callback(self, data):
        """
        Pointcloud subscriber callback
        """
        pc_raw = ros_np.point_cloud2.pointcloud2_to_array(data)

        # apply filter on pc array
        pc_filtered_arr = apply_filter(pc_raw)
        
        # convert back to pointcloud2 format
        f32 = np.float32
        pc_rec_arr = np.array(pc_filtered_arr, dtype=[('x', f32), ('y', f32), ('z', f32), ('rgb', f32)])
        
        # convert to pointcloud2 format for publishing
        pc_filtered = ros_np.point_cloud2.array_to_pointcloud2(pc_rec_arr, stamp=data.header.stamp, frame_id='map')      
        
        # publish filtered points
        print('Publish msg: {0} length: {1}'.format(self.msg_counter, np.size(pc_rec_arr)))
        self.pc_pub.publish(pc_filtered)
        
        self.msg_counter += 1


def apply_filter(raw_pc):
    """
    This function will perform the actual filtering and return a filtered pointcloud.
    """
    filtered_pc = raw_pc[1]
    
    sigma = 0.125
    step_size = 0.1 #0.2
    
    # make a grid
    grid_2d = make_grid(raw_pc, step_size)

    # apply gaussian filter
    filt_img = ndi.gaussian_filter(grid_2d, sigma=sigma)
    
    # decide which columns to keep by calculating mean
    mean = 368.68 # * 4 # np.mean(filt_img) * 4 # hardcoded based on data
    print('idx = {0} with mean = {1}'.format(sigma, mean))
    
    # find indices that are lower than global mean and set those to 0
    ind_zero = np.where(filt_img <= mean)
    filt_img[ind_zero] = 0
    
    # set all non-zero indices to 1
    ind_nonzero = np.nonzero(filt_img)
    filt_img[ind_nonzero] = 1
    
    # take x-y points
    x = raw_pc['x'].copy()
    y = raw_pc['y'].copy()
    
    # compute min-max grid edges
    x_min = -np.max(-x)
    y_min = -np.max(-y)
    
    # figure out for all our grid indices which points fall within the white squares
    for x_i in range(np.shape(filt_img)[0]):
    
        x_coord = np.floor(x_min) + step_size * x_i
        
        # loop over the y-axis
        for y_i in range(np.shape(filt_img)[1]):
            
            # break this iteration of for loop if we have a non-white square
            if filt_img[x_i, y_i] != 1.0:
                continue
            
            y_coord = np.ceil(y_min) + step_size * y_i
            
            result_x = np.where((x_coord < x) & (x <= x_coord + step_size))
            result_y = np.where((y_coord < y) & (y <= y_coord + step_size))
            
            ind_xy = np.intersect1d(result_x, result_y)
            
            # append the found points to our filtered pointcloud            
            filtered_pc = np.append(filtered_pc, raw_pc[ind_xy])           

    return filtered_pc
    

def Main():
    rospy.init_node('pc_filter', anonymous=True)
    pcf = pc_filter()
    print('start:')
    
    try:       
        # capture messages
        rospy.spin()   
        
    except KeyboardInterrupt:
        print('Shutting down')

if __name__ == '__main__':
    Main()


