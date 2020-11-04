# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Imports
import csv
from ast import literal_eval
import re
import numpy as np
import ResNet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# %%
"""
Both of the following functions import data. The output of both functions are a dictionary containing 5 arrays
    t: the array of av time points
    Q: the position values (q)
    P: the momentum values (p)
    T: the kinetic energy
    V: the potential energy
    
The data files contain data from 50 different trajectories, i.e. simulation of the path for a point with some 
initial position q0 and momentum p0. 

The function generate_data gives you the data from one of these data files, while the function concatenate
gives you the data from multiple trajectories at once. The default arguments of concatenate give you all the data
alltogether.

The folder project_2_trajectories must be placed in the same folder as your program to work. If the folder is in
some other location, the path for this location can be put into the string start_path.
"""
def generate_data(batch = 0):
    
    
    start_path = ""
    path = start_path+"project_2_trajectories/datalist_batch_" +str(batch)+".csv"
    with open(path,newline = "\n") as file:
        reader = csv.reader(file)
        datalist = list(reader)
    
    N = len(datalist)
    t_data = np.array([float(datalist[i][0]) for i in range(1,N)])
    Q1_data = [float(datalist[i][1]) for i in range(1,N)]
    Q2_data = [float(datalist[i][2]) for i in range(1,N)]
    Q3_data = [float(datalist[i][3]) for i in range(1,N)]
    P1_data = [float(datalist[i][4]) for i in range(1,N)]
    P2_data = [float(datalist[i][5]) for i in range(1,N)]
    P3_data = [float(datalist[i][6]) for i in range(1,N)]
    T_data = np.array([float(datalist[i][7]) for i in range(1,N)])
    V_data = np.array([float(datalist[i][8]) for i in range(1,N)])
                      
    Q_data = np.transpose(np.array([[Q1_data[i], Q2_data[i], Q3_data[i]] for i in range(N-1)]))
    P_data = np.transpose(np.array([[P1_data[i], P2_data[i], P3_data[i]] for i in range(N-1)]))
    
    return {"t": t_data, "Q": Q_data, "P": P_data, "T": T_data, "V": V_data}

def concatenate(batchmin=0, batchmax=50):
    dictlist = []
    for i in range(batchmin,batchmax):
        dictlist.append(generate_data(batch = i))
    Q_data = dictlist[0]["Q"]
    P_data = dictlist[0]["P"]
    T0 = dictlist[0]["T"]
    V0 = dictlist[0]["V"]
    tlist = dictlist[0]["t"]
    for j in range(batchmax-1):
        Q_data = np.hstack((Q_data, dictlist[j+1]["Q"]))
        P_data = np.hstack((P_data, dictlist[j+1]["P"]))
        T0 = np.hstack((T0, dictlist[j+1]["T"]))
        V0 = np.hstack((V0, dictlist[j+1]["V"]))
        tlist = np.hstack((tlist, dictlist[j+1]["t"]))
    return {"t": tlist, "Q": Q_data, "P": P_data, "T": T0, "V": V0}


# %% Read data
if __name__ == "__main__":

    max_batch_size = 5

    data = concatenate(0, max_batch_size)
    print("done")
    for key in data:
        print(f"data: {key}, shape: {data[key].shape}")


    Q = data['Q']
    P = data['P']
    V = data['V'][:,None]
    T = data['T'][:,None]

    Q_train, Q_test, V_train, V_test = train_test_split(Q.T, V, test_size=(1-1/max_batch_size/2), random_state=33)

    Q_train = Q_train.reshape(3, -1)
    Q_test = Q_test.reshape(3, -1)
    V_train = V_train.reshape(-1, 1)
    V_test = V_test.reshape(-1,1)

    K = 5      # Hidden layers
    tau = 0.001 # Learning parameter
    beta1 = 0.9
    beta2 = 0.999
    alpha = 0.01
    eps = 1e-8
    h = 0.1     # ?

    d0 = Q_test.shape[0]
    d = d0*2

    print(f"Num train data: {Q_train.shape}")

    # %% Train

    resnetV = ResNet.ResNet(K, d, tau, h, beta1, beta2, alpha, eps)
    resnetT = ResNet.ResNet(K, d, tau, h, beta1, beta2, alpha, eps)

    resnetV.setActivationFunction("sigmoid")
    resnetT.setActivationFunction("sigmoid")

    Q_scaled, Q_min, Q_max = resnetV.scale(Q_train)
    V_scaled, V_min, V_max = resnetV.scale(V_train)

    print(f"scaled data is: {Q_scaled.shape}, {V_scaled.shape}")

    resnet_dict = resnetV.train(Q_train, V_train, data_output=True, debug=True)

    print("done")
    # %% Plot results

    y = np.linspace(0, 1, num=1000)

    # upsilon = resnet.F(y)
    # grad, _, _ = resnet.scale_gradient(resnet.gradient(y))
    # o, _, _ = resnet.scale(f(np.arange(start,end,0.01)))
    # l,_,_ = resnet.scale_gradient(df(np.arange(start,end,0.01)))
    # upsilon = resnet.scale(resnet.F(y))[0]
    upsilon = resnetV.F(y)
    #upsilon = resnetV.rescale(upsilon, V_min, V_max)
    # l = df(np.linspace(start, end, num=len(y)))
    # grad = resnet.gradient(y)
    # # grad = resnet.rescale_gradient(grad, c_min, c_max)
    # o = f(np.linspace(start, end, num=len(y)))

    fig, ax = plt.subplots(figsize=(6,6), num=0)

    ax.plot(y, upsilon)

    fig2, ax2 = plt.subplots(figsize=(6,6), num=1)
    y = resnet_dict["Js"][20:]
    x = resnet_dict["iter"][20:]
    ax2.plot(x,y)
    ax2.set_ylabel(r'$J(\theta)$', rotation=0)

    ax2.set_xlabel("iterations")
    plt.show()

# %% Integrate P and Q

    Pint = P.copy()
    Qint = Q.copy()

    Pint[:,1:] = 0
    Qint[:,1:] = 0

