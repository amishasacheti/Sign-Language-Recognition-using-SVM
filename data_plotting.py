#display dataset in 3D
#STEP 1
import pandas as pd
import numpy as np
import math

from mpl_toolkits import mplot3d
%matplotlib inline
import matplotlib.pyplot as plt
data= np.loadtxt('D:\\final sem project\\dataset of coordinates\\1-hand_gestures\\afternoon_apurve_7.txt')
count=len(data)
bone_list = [[1, 2],[2,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],[2,11],[11,12],[12,13],[12,14],[13,15],[14,16],[15,17],[17,19],[16,18],[18,20]]

lis=[[2,3],[3,5],[2,5],[2,4],[4,6],[2,6]]
lis = np.array(lis) - 1
lis1=[[3,5],[5,7],[3,7],[4,6],[6,8],[4,8]]

lis1 = np.array(lis1) - 1
lis2=[[9,5],[5,7],[9,7],[10,6],[6,8],[10,8]]

lis2 = np.array(lis2) - 1
print(count)
bone_list = np.array(bone_list) - 1


def findangle(x1,x2,x3,y1,y2,y3,z1,z2,z3):
    cl=np.array([x1-x3,y1-y3,z1-z3])
    cr=np.array([x2-x3,y2-y3,z2-z3])
    modcl=np.linalg.norm(cl)
    modcr=np.linalg.norm(cr)
    m=modcl*modcr
    l=np.cross(cl,cr)
    n=l/m
    k=np.array([0,0,1])
    d=np.dot(n,k)
    modn=np.linalg.norm(n)
    modk=np.linalg.norm(k)
    a=d/(modn*modk)
    theta=math.acos(a)  #angle in radian
    theta1=(math.pi)-theta
    angle=(theta1*180)/math.pi  #angle in degree
    #print(angle)
    #print(theta1*180/math.pi)
    return angle


for i in range(count):
    #fig, ax = plt.subplots(1, figsize=(3, 8))

    
    fig = plt.figure()
    #ax = plt.axes(projection='3d')
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.set_title('Skeleton')
    #plt.title('Skeleton')
    #plt.xlim(-0.8,0.4)
    #plt.ylim(-1.5,1.5)

    x=data[i][0::3]
    y=data[i][1::3]
    z=data[i][2::3]
    #ax.scatter(x, y,s=40)
    ax.scatter3D(x, z,y,s=40)
        
    ax.set_xlim(-0.2,0.5)
    ax.set_ylim(-0.8,1.5)
    ax.set_zlim(0,2)
    for bone in bone_list:
        ax.plot([x[bone[0]], x[bone[1]]],[z[bone[0]], z[bone[1]]], [y[bone[0]], y[bone[1]]],'r')
        #ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]],[z[bone[0]], z[bone[1]]], 'r')
        #ax.plot([z[bone[0]], z[bone[1]]], [x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]],'r')
    
    
    #display dataset in 2D


for i in range(count):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax = fig.add_subplot(1, 2, 2, projection='3d')

    #ax.set_title('Skeleton')
    plt.title('Skeleton')
    plt.xlim(-0.8,0.4)
    plt.ylim(-1.5,1.5)

    x=data[i][0::3]
    y=data[i][1::3]
    z=data[i][2::3]
    ax.scatter(x, y,s=40)
    #ax.scatter3D(x, y, z,s=40)
        
    #ax.set_xlim(-0.8,-0.1)
    #ax.set_ylim(-0.5,1.5)
    #ax.set_zlim(-3,4)
    for bone in bone_list:
        ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]], 'r')
        
