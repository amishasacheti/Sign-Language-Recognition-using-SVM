#affine transformation 2D+featureization

import math
import numpy as np

data= np.loadtxt('D:\\final sem project\\dataset of coordinates\\1-hand_gestures\\afternoon_apurve_7.txt')
count=len(data)
bone_list = [[1, 2],[2,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],[2,11],[11,12],[12,13],[12,14],[13,15],[14,16],[15,17],[17,19],[16,18],[18,20]]
bone_list = np.array(bone_list) - 1

def distance(x1,x2,y1,y2):
    dist=math.sqrt(((x2-x1)**2)+((y2-y1)**2))
    return dist


def findangle(x1,x2,x3,y1,y2,y3,z1,z2,z3,k):
    cl=np.array([x1-x3,y1-y3,z1-z3])
    cr=np.array([x2-x3,y2-y3,z2-z3])
    modcl=np.linalg.norm(cl)
    modcr=np.linalg.norm(cr)
    m=modcl*modcr
    l=np.cross(cl,cr)
    n=l/m
    #k=[0,0,1]
    d=np.dot(n,k)
    modn=np.linalg.norm(n)
    modk=np.linalg.norm(k)
    a=d/(modn*modk)
    theta=math.acos(a)  #angle in radian
    theta1=(math.pi)-theta
    angle=theta*180/math.pi  #angle in degree
    #print(theta1*180/math.pi)
    return theta1

def solve_affine( h,A ):
    
    # add ones on the bottom of x 
    x = np.vstack((h,[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
    #x = np.vstack((h,[1]))
   
    #return lambda x: (A*np.vstack((np.matrix(x).reshape(3,1),1)))[0:3,:]
    transform=np.dot(A,x)
    #print(transform)
    return np.dot(A,x)[0:3,:]
    
for i in range(count):
    h=0
    a=0
    x1=data[i][6]
    y1=data[i][7]
    z1=data[i][8]
    x2=data[i][9]
    y2=data[i][10]
    z2=data[i][11]
    x3=data[i][30]
    y3=data[i][31]
    z3=data[i][32]
    
    
    #k=[0,1,0]
    k1=[0,0,1]
    #k2=[1,0,0]
    #theta=findangle(x1,x2,x3,y1,y2,y3,z1,z2,z3,k)
    theta1=findangle(x1,x2,x3,y1,y2,y3,z1,z2,z3,k1)
    #theta2=findangle(x1,x2,x3,y1,y2,y3,z1,z2,z3,k2)
    #l=math.cos(theta)
    #m=math.sin(theta)
    
    l1=math.cos(theta1)
    m1=math.sin(theta1)
    
    #l2=math.cos(theta2)
    #m2=math.sin(theta2)
    
    #rotation_x=np.array([[l,0,0,0],[0,l,-m,0],[0,m,l,0],[0,0,0,1]])
    #inverse_x=np.array([[l,0,0,0],[0,l,m,0],[0,-m,l,0],[0,0,0,1]])
    rotation_y=np.array([[l1,0,m1,0],[0,1,0,0],[-m1,0,l1,0],[0,0,0,1]])
    inverse_y=np.array([[l1,0,-m1,0],[0,1,0,0],[m1,0,l1,0],[0,0,0,1]])
    #rotation_z=np.array([[l2,-m2,0,0],[m2,l2,0,0],[0,0,1,0],[0,0,0,1]])
    
    translation=np.array([[1,0,0,-x3],[0,1,0,-y3],[0,0,1,-z3],[0,0,0,1]])
    inverse_translation=np.array([[1,0,0,x3],[0,1,0,y3],[0,0,1,z3],[0,0,0,1]])
    
   # p=np.dot( inverse_translation, inverse_x)
   # q=np.dot(p,inverse_y)
   # r=np.dot(q,rotation_z)
   # s=np.dot(r,rotation_y)
   # t=np.dot(s,rotation_x)
   # u=np.dot(t,translation)
    u=np.dot(translation,rotation_y)
    #for j in range(0,60,3):
        #a=data[i][j]
        #b=data[i][j+1]
        #c=data[i][j+2]
        #A = np.array([[a, b, c]])
        #transform=np.dot(rotation_matrix,(np.transpose(A)))
        #h=np.append(h,transform)
   
    h=np.array([0,0,0])
    h=np.transpose(h)
    #h=np.delete(h,0)
    for j in range(0,60,3):
        hell=np.array([data[i][j],data[i][j+1],data[i][j+2]])
        hell=np.transpose(hell)
        h=np.column_stack((h, hell))
    
    h=np.delete(h,0,1)
    ho=solve_affine(h,u)
    m=0
    for j in range(0,20):
        x=ho[:,[j]]
        l=np.transpose(x)
        m=np.append(m,l)
    
    m=np.delete(m,0)
    
    
     # let feature 1 be triangle between sc, sl, el (2,3,5)
    c1x=(m[3]+m[6]+m[12])/3;
    c1y=(m[4]+m[7]+m[13])/3;
    #c1z=(m[5]+m[8]+m[14])/3;
    
    # let feature 2 be triangle between sc, sr, er (2,4,6)
    c2x=(m[3]+m[9]+m[15])/3;
    c2y=(m[4]+m[10]+m[16])/3;
    #c2z=(m[5]+m[11]+m[17])/3;
    
    # let feature 3 be triangle between  sl, el, wl (3,5,7)
    c3x=(m[6]+m[12]+m[18])/3;
    c3y=(m[7]+m[13]+m[19])/3;
    #c3z=(m[8]+m[14]+m[20])/3;
    
    # let feature 4 be triangle between sr, er,wr (4,6,8)
    c4x=(m[9]+m[15]+m[21])/3;
    c4y=(m[10]+m[16]+m[22])/3;
    #c4z=(m[11]+m[17]+m[23])/3;
    
     # let feature 5 be triangle between  el, wl,tl (5,7,9)
    c5x=(m[12]+m[18]+m[24])/3;
    c5y=(m[13]+m[19]+m[25])/3;
    #c5z=(m[14]+m[20]+m[26])/3;
    
    # let feature 6 be triangle between  er,wr,tr (6,8,10)
    c6x=(m[15]+m[21]+m[27])/3;
    c6y=(m[16]+m[22]+m[28])/3;
    #c6z=(m[17]+m[23]+m[29])/3;
    
    #finding euclidean distance
    e1=distance(m[30],c1x,m[31],c1y)
    print('e1: ',e1)
    e2=distance(m[30],c2x,m[31],c2y)
    print('e2: ',e2)
    e3=distance(m[30],c3x,m[31],c3y)
    print('e3: ',e3)
    e4=distance(m[30],c4x,m[31],c4y)
    print('e4: ',e4)
    e5=distance(m[30],c5x,m[31],c5y)
    print('e5: ',e5)
    e6=distance(m[30],c6x,m[31],c6y)
    print('e6: ',e6)
    
    
   

    fig, ax = plt.subplots(1, figsize=(8, 8))
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax = fig.add_subplot(1, 2, 2, projection='3d')

    #ax.set_title('Skeleton')
    plt.title('Skeleton')
    #plt.xlim(-0.25,1.5)
    #plt.ylim(-1.5,1.5)
    #plt.xlim(0.2,0.8)
    
    x=m[0::3]
    y=m[1::3]
    z=m[2::3]
    
    ax.scatter(x, y,s=40)
    ax.scatter(c1x, c1y,s=40)
    ax.scatter(c2x, c2y,s=40)
    ax.scatter(c3x, c3y,s=40)
    ax.scatter(c4x, c4y,s=40)
    ax.scatter(c5x, c5y,s=40)
    ax.scatter(c6x, c6y,s=40)
    #ax.scatter3D(z, x, y,s=40)
    ax.plot([c1x,m[30]],[c1y,m[31]],'deepskyblue')
    ax.plot([c2x,m[30]],[c2y,m[31]],'deepskyblue')
    ax.plot([c3x,m[30]],[c3y,m[31]],'deepskyblue')
    ax.plot([c4x,m[30]],[c4y,m[31]],'deepskyblue')
    ax.plot([c5x,m[30]],[c5y,m[31]],'deepskyblue')
    ax.plot([c6x,m[30]],[c6y,m[31]],'deepskyblue')
    #ax.set_xlim(-0.8,-0.1)
    #ax.set_ylim(-0.5,1.5)
    #ax.set_zlim(-3,4)
    for bone in bone_list:
        #ax.plot([z[bone[0]], z[bone[1]]], [x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]],'r')
        ax.plot([x[bone[0]], x[bone[1]]], [y[bone[0]], y[bone[1]]],'r')
    
