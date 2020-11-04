#################################################################################################
### WARNING: This code takes approximately 250s to run, as criteria for convergence is strict ###
#################################################################################################

import ResNet
from project_2_data_acquisition import concatenate
from integrate import integrate, symplectic_euler, stormer_verlet

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

# Retrieve Hamiltonian data
max_batch_size = 5
data = concatenate(0, max_batch_size)

t = data['t']
Q = data['Q']
P = data['P']
V = data['V'][:,None]
T = data['T'][:,None]

d0, data_points = Q.shape

# Divide data into training set and test set
data_slice = int(data_points/5)
t = t[:data_slice]

Q_train = Q[:,data_slice:]
Q_test = Q[:,:data_slice]
V_train = V[data_slice:]
V_test = V[:data_slice]

P_train = P[:,data_slice:]
P_test = P[:,:data_slice]
T_train = T[data_slice:]
T_test = T[:data_slice]

# Network parameters
K = 5
d = 2*d0
h = 0.1
tau = 0.001

# Initialize networks
resnetV = ResNet.ResNet(K,d,tau,h)
resnetT = ResNet.ResNet(K,d,tau,h)

# Train networks, this is time consuming. Increase J_threshold for faster convergence, but less accurate results
resnetV_dict = resnetV.train(Q_train, V_train, J_threshold=0.05, debug=True, data_output=True)
resnetT_dict = resnetT.train(P_train, T_train, J_threshold=0.05, debug=True, data_output=True)

# Forward sweeps
upsilon_T = resnetT.F(P_test)
upsilon_V = resnetV.F(Q_test)

# Plotting of network output against true values
plt.figure(figsize=(6,6))
plt.plot(range(P_test.shape[1]),upsilon_T, label="network")
plt.plot(range(P_test.shape[1]), T_test, label="true")
plt.ylabel("T(p)")
plt.xlabel("data points")
plt.legend()

plt.figure(figsize=(6,6))
plt.plot(range(Q_test.shape[1]),upsilon_V, label="network")
plt.plot(range(Q_test.shape[1]), V_test, label="true")
plt.ylabel("V(q)")
plt.xlabel("data points")
plt.legend()

# Gradients of networks to be used for integration
def Vgrad(q): return resnetV.gradient(q)
def Tgrad(p): return resnetT.gradient(p)

# Integration parameters
h = np.average(np.diff(t))
p0 = P_test[:,0]
q0 = Q_test[:,0]
T = t[-1]

# Integration
Pinted, Qinted = integrate(p0, q0, h, Tgrad, Vgrad, T, symplectic_euler)

# Plotting of final results
plt.figure(figsize=(6,6))
plt.subplot(311)
plt.plot(Pinted[:,0],label="network")
plt.plot(P_test[0,:], label="true")
plt.ylabel(r'$p_1$')

plt.subplot(312)
plt.plot(Pinted[:,1],label="network")
plt.plot(P_test[2,:], label="true")
plt.ylabel(r'$p_2$')

plt.subplot(313)
plt.plot(Pinted[:,0],label="network")
plt.plot(P_test[0,:], label="true")
plt.ylabel(r'$p_3$')

plt.legend()

plt.figure(figsize=(6,6))
plt.subplot(311)
plt.plot(Qinted[:,0],label="network")
plt.plot(Q_test[0,:], label="true")
plt.ylabel(r'$q_1$')

plt.subplot(312)
plt.plot(Qinted[:,1],label="network")
plt.plot(Q_test[2,:], label="true")
plt.ylabel(r'$q_2$')

plt.subplot(313)
plt.plot(Qinted[:,0],label="network")
plt.plot(Q_test[0,:], label="true")
plt.ylabel(r'$q_3$')

plt.legend()

plt.show()