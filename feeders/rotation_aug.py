import os
import sys
import numpy as np
import math
import random


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)


def get_rot_mat_random(theta):
    # angles = theta.reshape(-1)
    angles=theta
    basis = rotate_basis(np.eye(3), angles)
    return basis


def random_rotation(data_numpy, theta):
    '''
    theta=np.array([th1,th2,th3]) in rad.
    data_numpy: (C,T,V,M)
    theta=np.random.uniform(-10,10,size=3)/180.0*2*np.pi
    '''
    C,T,V,M=data_numpy.shape
    data_numpy= np.transpose(data_numpy, [3,1,2,0]).reshape(-1,C)

   
    rot_mat=get_rot_mat_random(theta)


    data_numpy= np.dot(data_numpy, rot_mat.T)
    data_numpy= data_numpy.reshape([M,T,V,C])
    data_numpy= np.transpose(data_numpy, [3,1,2,0])

    return data_numpy

def rotate_basis(local3d, angles):
    """
    Rotate local rectangular coordinates from given view_angles.

    :param local3d: numpy array. Unit vectors for local rectangular coordinates's , shape (3, 3).
    :param angles: tuple of length 3. Rotation angles around each axis.
    :return:
    """
    cx, cy, cz = np.cos(angles)
    sx, sy, sz = np.sin(angles)

    x = local3d[0]
    x_cpm = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype='float')
    x = x.reshape(-1, 1)
    mat33_x = cx * np.eye(3) + sx * x_cpm + (1.0 - cx) * np.matmul(x, x.T)

    mat33_z = np.array([
        [cz, sz, 0],
        [-sz, cz, 0],
        [0, 0, 1]
    ], dtype='float')

    local3d = local3d @ mat33_x.T @ mat33_z
    return local3d


def my_random_rot(data_numpy, theta_val):
    theta=np.random.uniform(-theta_val,theta_val,size=3)/180.0*2*np.pi
    data_numpy=random_rotation(data_numpy, theta)
    return data_numpy


def my_random_rot_fixed(data_numpy, theta):
    # theta=np.random.uniform(-theta_val,theta_val,size=3)/180.0*2*np.pi
    data_numpy=random_rotation(data_numpy, theta)
    return data_numpy











def rot_align_v1(data, zaxis=[0, 1], xaxis=[8, 4]):

    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C
    # m,t,v,c
    skeleton = s[0]

    joint_bottom = skeleton[0, 0, zaxis[0]]
    joint_top = skeleton[0, 0, zaxis[1]]
    axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
    angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
    matrix_z = rotation_matrix(axis, angle)

    skeleton = np.transpose(skeleton, [3,0,1,2]).reshape(C,M*T*V)
    skeleton = np.dot(matrix_z, skeleton).reshape(C,M,T,V)
    skeleton = np.transpose(skeleton, [1,2,3,0])


    joint_rshoulder = skeleton[0, 0, xaxis[0]]
    joint_lshoulder = skeleton[0, 0, xaxis[1]]
    axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
    angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
    matrix_x = rotation_matrix(axis, angle)

    skeleton = np.transpose(skeleton, [3,0,1,2]).reshape(C,M*T*V)
    skeleton = np.dot(matrix_x, skeleton).reshape(C,M,T,V)
    skeleton = np.transpose(skeleton, [1,2,3,0])

    # (m,t,v,c)
    s = skeleton[None]
    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data







def rot_align_v0(data, zaxis=[0, 1], xaxis=[8, 4]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    print('parallel the bone between hip(jpt 0) and spine(jpt 1) of the first person to the z axis')
    for i_s, skeleton in enumerate((s)):
        if skeleton.sum() == 0:
            continue
        joint_bottom = skeleton[0, 0, zaxis[0]]
        joint_top = skeleton[0, 0, zaxis[1]]
        axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_z, joint)

    print(
        'parallel the bone between right shoulder(jpt 8) and left shoulder(jpt 4) of the first person to the x axis')
    for i_s, skeleton in enumerate((s)):
        if skeleton.sum() == 0:
            continue
        joint_rshoulder = skeleton[0, 0, xaxis[0]]
        joint_lshoulder = skeleton[0, 0, xaxis[1]]
        axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = rotation_matrix(axis, angle)
        for i_p, person in enumerate(skeleton):
            if person.sum() == 0:
                continue
            for i_f, frame in enumerate(person):
                if frame.sum() == 0:
                    continue
                for i_j, joint in enumerate(frame):
                    s[i_s, i_p, i_f, i_j] = np.dot(matrix_x, joint)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data


