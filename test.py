# -- coding: utf-8 --
"""
Created on Thu Dec  3 18:33:29 2020

@author: Antika
"""
import re
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import scipy
import copy
import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
from scipy.spatial import distance
from matplotlib import cm
import seaborn as sns




#
def read_file_original(file_path):
    a = []
    with open(file_path) as f:
        content = f.readlines()
        for line in content:
            x = float(re.split('\s+', line)[0])
            y = float(re.split('\s+', line)[1])
            z = float(re.split('\s+', line)[2])

            b = np.array([x, y, z])
            a.append(b)

    data = np.array(a)
    return data


def Optimal_Rotation_translation(A, B):
    # get number of dimensions
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    new_A = A - centroid_A
    new_B = B - centroid_B

    # SVD
    W = np.dot(new_A.T, new_B)
    U, D, Vt = np.linalg.svd(W)
    # Rotation Matrix
    R = np.dot(Vt.T, U.T)
    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)
    # homogeneous transformation
    H_T = np.identity(m + 1)

    H_T[:m, :m] = R
    H_T[:m, m] = t
    #
    return H_T, R, t


def nearest_neighbor_kd_tree(src, dst):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(dst)
    distances, indices = nbrs.kneighbors(src, return_distance=True)
    return distances, indices


# =============================================================================
# #DEFINE ICP
# =============================================================================

def icp(A, B, tolerance=0.000001):
    R_stack=[]
    t_stack=[]
        

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    prev_error = 0
    start = time.time()
    for i in range(1000):
        print (i)
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor_kd_tree(src[:m, :].T, dst[:m, :].T)
        src_indx = [index for index in range(0, len(indices))]
        src1 = np.array([list(src.T[i]) for i in src_indx])
        src1 = src1.T
        dst1 = np.array([dst.T[i][0] for i in indices])
        dst1 = dst1.T
        #        print (np.mean(distances))

        # compute the transformation between the current source and nearest destination points
        H_T, R, t = Optimal_Rotation_translation(src1[:m, :].T, dst1[:m, :].T)
        R_stack.append(R.tolist())
        t_stack.append(t.tolist())
        #
        # update the current source
        src = np.dot(H_T, src)
        #        print (prev_error-np.mean(distances))

        if np.abs(prev_error - np.mean(distances)) < tolerance:
            print("satisfied")
            break
        prev_error =np.mean(distances)
    end = time.time()
    run_time_icp=end-start
    
    H_T,R,t = Optimal_Rotation_translation(A, src[:m,:].T)
    
    project_points = np.ones((len(A), 4))
    project_points[:,0:3] = A

        # Transform C
    project_points = np.dot(H_T, project_points.T).T    
  
    print(f"Runtime of the program for ICP is is {end - start}")
    return project_points,H_T, R, t, distances, i,R_stack,t_stack,run_time_icp
  

# =============================================================================
# DEFINE TR_ICP
# =============================================================================

def tr_icp(A, B, tolerance=0.000001):
    R_stack=[]
    t_stack=[]
    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)
    #    print (src.T)
    prev_error = 0
    start = time.time()

    for i in range(1000):
        print (i)
        #        find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor_kd_tree(src[:m, :].T, dst[:m, :].T)
        percent = int(len(src[:m, :].T)*0.75)

        #
        src_indx = [index for index in range(0, len(indices))]

        zipped = list(zip(src_indx, indices.ravel(), distances.ravel()))
        #        zipped1=list(zip(src_indx[0:10],indices[:10],distances[0:10]))

        res = sorted(zipped, key=lambda x: x[2])
        sorted_pair_dis = res[0:percent]
        src_sorted = [m[0] for m in sorted_pair_dis]
        #        print (src_sorted)
        dst_sorted = [m[1] for m in sorted_pair_dis]
        #        print (dst_sorted)
        distance_sorted_distance = [m[2] for m in sorted_pair_dis]
        #        print (distance_sorted_distance)
        src1 = np.array([list(src.T[i]) for i in src_sorted])
        dst1 = np.array([list(dst.T[i]) for i in dst_sorted])
        #        print (src1.T.shape)
        src1 = np.delete(src1, 3, 1)
        dst1 = np.delete(dst1, 3, 1)
        
        H_T, R, t = Optimal_Rotation_translation(src1, dst1)
        
        R_stack.append(R.tolist())
        t_stack.append(t.tolist())

        ##        # update the current source
        src = np.dot(H_T, src)
        if np.abs(prev_error - np.mean(distance_sorted_distance)) < tolerance:
            print("satisfied")
            break
        prev_error =np.mean(distance_sorted_distance)
    end = time.time()
    
    run_time_tr_icp=end-start
    print(f"Runtime of the program for tr_icp is {end - start}")
    
    
    H_T,R,t = Optimal_Rotation_translation(A, src[:m,:].T)
    
    project_points = np.ones((len(A), 4))
    project_points[:,0:3] = A

        # Transform C
    project_points = np.dot(H_T, project_points.T).T    
  
    return project_points,H_T, R, t, distances, i,R_stack,t_stack,run_time_tr_icp

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

# =============================================================================
# #ADD gaussian noise
# =============================================================================

def gaussian_noise(data,mu,sigma):
    data_h, data_w = data.shape

#    mu, sigma = 1, 0.01
    # creating a noise with the same dimension as the dataset (2,2) 
    noise_data = np.random.normal(mu, sigma, [data_h, data_w])
    X, Y, Z = noise_data.T[0], noise_data.T[1], noise_data.T[2]

    print ("noise_data :", noise_data)
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(X,Y, Z, color = "green")
    plt.title("simple 3D scatter plot")    
    data = data + noise_data
    return data
    
def Affine_t(value):
    alpha = math.radians(value)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    Rx = np.array(((1, 0, 0), (0, cos_alpha, -sin_alpha), (0, sin_alpha, cos_alpha)))   
    Ry = np.array(((cos_alpha, 0, sin_alpha), (0, 1, 0), (-sin_alpha, 0, cos_alpha)))
    Rz = np.array(((cos_alpha, -sin_alpha, 0), (sin_alpha, cos_alpha, 0), (0, 0, 1)))

    R_init = np.dot(np.dot(Rz, Ry), Rx)
    
    Translation_vec = np.array([1, 1, 1])

    return R_init,Translation_vec


if __name__ == "__main__":
    
# =============================================================================
# READ OR GENERATE 3D POINT CLOUD
# =============================================================================
    
##    Create synthetically 3d points
    N = 1000                                    # number of random points in the dataset
    dim = 3                                     # number of dimensions of the points
#    
    A = np.random.rand(N, dim)

    total_time = 0

    for i in range(1000):

        B = np.copy(A)

        # Translate
        B =np.random.rand(dim)*5+B

        # Rotate
        R,t = Affine_t(30)
        B = np.dot(R, B.T).T

        # Add noise
        B += np.random.randn(N, dim) * 0.02
        np.random.shuffle(B)


##        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(A)
    o3d.io.write_point_cloud("synthetic_data_A.xyz", pcd)
##    
#            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(B)
    o3d.io.write_point_cloud("synthetic_data_B.xyz", pcd)
#Read Files

#    
##    
#    A=read_file_original("bunny_a.xyz")
#    B=read_file_original("bunny_b.xyz")
####    
#    A=read_file_original("LionScan1.xyz")
#    B=read_file_original("LionScan2_new.xyz")
#
#    A=read_file_original("data_src_A_new.xyz")
#    B=read_file_original("data_src_B.xyz")

    

    B = gaussian_noise(B,0,0.01)    
#
##    
    
    
# =============================================================================
# EVALUATION
# =============================================================================
##
    R_init,Translation_vec = Affine_t(30)

    
    data = read_file_original("data_src_A.xyz")
    #   data=read_file_original("point_cloud_a_house.xyz")
#    data=gaussian_noise(data,mu,sigma)
    
    transformed_data = np.dot(data, R_init)+Translation_vec
#    transformed_data=gaussian_noise(transformed_data,0,0.01)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_data)
    o3d.io.write_point_cloud("transformed_data_src_A.xyz", pcd)
    
    transformed_data_b=np.dot(transformed_data, R_init)+Translation_vec
    

    transformed_data_icp, transformed_data_H_T, transformed_data_R, transformed_data_t, distances,i,R_stack_icp,t_stack_icp,run_time_icp= icp(data,transformed_data_b)
    transformed_data_icp = np.delete(transformed_data_icp, 3, 1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_data_icp)
    o3d.io.write_point_cloud("transformed_data__icp_src_A.xyz", pcd)
                                                                                                 
    
#    
    transformed_data_tr_icp, transformed_data_tr_H_T, transformed_data_tr_R, transformed_data_tr_t, distances, i,R_stack_tr_icp,t_stack_tr_icp,run_time_tr_icp= tr_icp(data,transformed_data)
    transformed_data_icp = np.delete(transformed_data_tr_icp, 3, 1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_data_icp)
    o3d.io.write_point_cloud("transformed_data_TR_icp_src_A.xyz", pcd)

    R_stack_for_error=[]
    t_stack_for_error=[]
    
    for i in R_stack_icp:
        R=np.array(i)
        R_=R*R_init.T
        
        print (R)
        geodist=(np.log2(R_))
        geodist=np.nan_to_num(np.nan_to_num(geodist))
        print (geodist)
        euler_angle=np.nan_to_num(rot2eul(geodist))
        
        euler_angle_in_degree=(euler_angle*180)/math.pi
#        print (euler_angle_in_degree)
        R_stack_for_error.append(euler_angle_in_degree)
    R_stack_for_error=np.array(R_stack_for_error)
#       
    X, Y, Z = R_stack_for_error.T[0], R_stack_for_error.T[1], R_stack_for_error.T[2]
    X=X.tolist()
    Z=Z.tolist()
    Y=Y.tolist()
    for i in t_stack_icp:
        
        dst = distance.euclidean(Translation_vec,i)
        t_stack_for_error.append(dst)
        
        print (dst)  
        
    plt.plot(range(0,len(t_stack_for_error)),t_stack_for_error)
    plt.xlabel("ieration")
    plt.ylabel("euclidean")
    plt.show()
    
    plt.plot(range(0,len(X)),X,color="red",label="alpha")
    plt.xlabel("iterations")
    plt.ylabel("degree")
    plt.legend()

    plt.show()
    
    plt.plot(range(0,len(Y)),Y,color="blue",label="beta")
    plt.xlabel("iterations")
    plt.ylabel("degree")
    plt.legend()
    plt.show()
    
    plt.plot(range(0,len(Z)),Z,label="gamma")
    plt.xlabel("iterations")
    plt.ylabel("degree")
    plt.legend()
    plt.show()
#    
#    

# =============================================================================
# #ICP
# =============================================================================

    src,H_T_icp,R_icp,t_icp,dist_icp,i_icp,stack_icp_r,stack_icp_t,run_time_icp=icp(A,B)
    print (src)      
    src = np.delete(src,3,1)        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(src)
    o3d.io.write_point_cloud("output_icp.xyz", pcd)
###    
## =============================================================================
## #Trimmed_icp
##     
## =============================================================================
    src_tr_icp,H_T_tr_icp,R_tr_icp,t_tr_icp,dist_tr_icp,i_tr_cp,stack_R_tr_icp,stack_t_tr_icp,run_time_tr=tr_icp(A,B)
#    src=src_tr_icp.T
    src_tr_icp = np.delete(src_tr_icp,3,1)  
    print (src_tr_icp)      
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(src_tr_icp)
    o3d.io.write_point_cloud("output_tr_icp.xyz", pcd)


    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    l= ["ICP","TR_ICP"]
    cmp = [run_time_icp,run_time_tr]
    ax.bar(l,cmp)
    plt.show()
    
#   
#
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    l= ["iteration icp","iteration TR_ICP"]
    cmp = [i_icp,i_tr_cp]
    ax.bar(l,cmp,color="red")
    plt.show()