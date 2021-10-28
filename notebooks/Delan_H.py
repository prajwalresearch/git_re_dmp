#!/usr/bin/env python
# coding: utf-8

# In[1]:


from posixpath import join
from cvxopt import solvers
import random
import matplotlib.pyplot as plt
from numpy.lib.function_base import average
import torch
from autograd import grad
from autograd import jacobian
import autograd.numpy as np
import autograd.numpy as jnp
import scipy.optimize as optim
from scipy.optimize import minimize, Bounds,LinearConstraint
from scipy.optimize import LinearConstraint,NonlinearConstraint
from scipy.optimize import BFGS
from  autograd.numpy import cos,sin
import pdb
#import pybulletIK_main as ik_solver
#from Metric_Ik_Analytic_witthOrient import traj_cost
t = 0.02
q_prev = None

device = 'cpu'
model = torch.load('models/model_750_model_epoch_20000.pth', map_location=torch.device('cpu'))  # loaded trained model #TODO
q_dim = 6  # q_dim is the dimension of joint space
q_dim_changed = int(0.5 * q_dim)

#Weight for cost function  
w_des_vel = 0.003
weight_orient = 0.2
#Desired final Orientation of end effector
roll_des= -3.141105126296926
pitch_des= 0.00046035505135551175
yaw_des = -2.355906195444897
orient_desired  = np.asarray([ roll_des , pitch_des , yaw_des ])

filesaveloc = 'trajc/cart_pose_3_obs_'

#value function defnation
weight = []
for key in (model.keys()):
    # print(key)
    weight.append(model[key].cpu().numpy())  # load weight and bias


def leaky_relu(z):
    return np.maximum(0.01 * z, z)


def softplus(z, beta=1):
    return (1 / beta) * np.log(1 + np.exp(z * beta))


def assemble_lower_triangular_matrix(Lo, Ld):
    Lo = Lo.squeeze(0)
    Ld = Ld.squeeze(0)

    assert (2 * Lo.shape[0] == (Ld.shape[0] ** 2 - Ld.shape[0]))
    diagonal_matrix = np.identity(len(Ld)) * np.outer(np.ones(len(Ld)), Ld)

    L = np.tril(np.ones(diagonal_matrix.shape)) - np.eye(q_dim_changed)

    # Set off diagonals

    L = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]) * Lo.reshape(3)[0] + np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]) *         Lo.reshape(3)[1] + np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]) * Lo.reshape(3)[2]
    # Add diagonals
    L = L + diagonal_matrix
    return L


def value(x1,goal):
    global weight
    fc1_w = weight[0]
    fc1_b = weight[1]
    fc2_w = weight[2]
    fc2_b = weight[3]
    fc_Ld_w = weight[4]
    fc_Ld_b = weight[5]
    fc_Lo_w = weight[6]
    fc_Lo_b = weight[7]
    #pdb.set_trace()
    net_input = np.concatenate([np.squeeze(x1), np.squeeze(goal)], axis=0)
    net_input = np.array([net_input])

    z1 = np.dot(net_input, fc1_w.transpose()) + fc1_b
    hidden1 = leaky_relu(z1)
    z2 = np.dot(hidden1, fc2_w.transpose()) + fc2_b
    hidden2 = leaky_relu(z2)
    hidden3 = np.dot(hidden2, fc_Ld_w.transpose()) + fc_Ld_b
    Ld = softplus(hidden3)
    Lo = np.dot(hidden2, fc_Lo_w.transpose()) + fc_Lo_b
    L = assemble_lower_triangular_matrix(Lo, Ld)

    H = L @ L.transpose() + 1e-9 * np.eye(3)
    return H

if __name__ == "__main__":
    print("hello")
    x0 = np.array([0, 0,0], dtype=np.float64)
    g = np.array([1, 1,0], dtype=np.float64)
    v = value(x0,g)
    print(v)