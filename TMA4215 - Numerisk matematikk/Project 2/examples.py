#####################################################
### Run python3 examples.py to run code at bottom ###
#####################################################

import ResNet

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def function1(samples=1000):
    d0 = 1
    d = 2

    start = -2
    end = 2

    f = lambda y : (0.5*y**2)
    df = lambda y : y

    Y0_train = ((end - start) * np.random.random_sample(samples) + start).reshape((d0,samples))
    c_train = f(Y0_train).reshape((samples,1))

    test_samples = int(samples/10)
    Y0_test = ((end - start) * np.random.random_sample(test_samples) + start).reshape((d0,test_samples))
    c_test = f(Y0_test).reshape((test_samples,1))

    return {"d0":d0, "d":d, "f":f, "Y0_train":Y0_train, "Y0_test":Y0_test, "c_train":c_train, "c_test":c_test, "start":start, "end":end, "df":df}

def function2(samples=1000):
    d0 = 1
    d = 2

    start = -np.pi/3
    end = np.pi/3

    f = lambda y : (1-np.cos(y))
    df = lambda y: (np.sin(y))

    Y0_train = ((end - start) * np.random.random_sample(samples) + start).reshape((d0,samples))
    c_train = f(Y0_train).reshape((samples,1))

    test_samples = int(samples/10)
    Y0_test = ((end - start) * np.random.random_sample(test_samples) + start).reshape((d0,test_samples))
    c_test = f(Y0_test).reshape((test_samples,1))

    return {"d0":d0, "d":d, "f":f, "Y0_train":Y0_train, "Y0_test":Y0_test, "c_train":c_train, "c_test":c_test, "start":start, "end":end, "df":df}

def function3(samples=1000):
    d0 = 2
    d = 4

    start = np.array([-2,-2])
    end = np.array([2,2])

    f = lambda y : 0.5*(y[0]**2+y[1]**2)

    Y0_train = ((end - start) * np.random.random_sample((samples,d0)) + start).reshape((d0,samples))
    c_train = f(Y0_train).reshape((samples,1))

    test_samples = int(samples/10)
    Y0_test = ((end - start) * np.random.random_sample((test_samples,d0)) + start).reshape((d0,test_samples))
    c_test = f(Y0_test).reshape((test_samples,1))

    return {"d0":d0, "d":d, "f":f, "Y0_train":Y0_train, "Y0_test":Y0_test, "c_train":c_train, "c_test":c_test, "start":start, "end":end}

def function4(samples=1000):
    d0 = 2
    d = 4

    start = np.array([-2,-2])
    end = np.array([2,2])

    f = lambda y : -1/(np.sqrt(y[0]**2 + y[1]**2))

    Y0_train = ((end - start) * np.random.random_sample((samples,d0)) + start).reshape((d0,samples))
    c_train = f(Y0_train).reshape((samples,1))

    test_samples = int(samples/10)
    Y0_test = ((end - start) * np.random.random_sample((test_samples,d0)) + start).reshape((d0,test_samples))
    c_test = f(Y0_test).reshape((test_samples,1))

    return {"d0":d0, "d":d, "f":f, "Y0_train":Y0_train, "Y0_test":Y0_test, "c_train":c_train, "c_test":c_test, "start":start, "end":end}



if __name__ == "__main__":
    # Can change between function1(), function2(), function3() and function4() to see examples
    data_dict = function1(samples=1000)
    Y0_train = data_dict["Y0_train"]
    Y0_test = data_dict["Y0_test"]
    c_train = data_dict["c_train"]
    c_test = data_dict["c_test"]
    d = data_dict["d"]
    d0 = data_dict["d0"]
    f = data_dict["f"]
    start = data_dict["start"]
    end = data_dict["end"]

    K = 5       # Hidden layers
    tau = 0.001 # Learning parameter
    h = 0.1     # Layer weight

    # Initialize network
    resnet = ResNet.ResNet(K,d,tau,h)

    # Scale data
    Y0_train_scaled, Y0_train_min, Y0_train_max = resnet.scale(Y0_train)
    Y0_test_scaled, Y0_test_min, Y0_test_max = resnet.scale(Y0_test)
    c_train_scaled, c_train_min, c_train_max = resnet.scale(c_train)
    c_test_scaled, c_test_min, c_test_max = resnet.scale(c_test)

    # Start training
    resnet_dict = resnet.train(Y0_train_scaled, c_train_scaled, J_threshold=1, data_output=True, debug=True)

    # Forward sweeps on test data
    upsilon_scaled = resnet.F(Y0_test_scaled)
    upsilon = resnet.rescale(upsilon_scaled, c_test_min, c_test_max)

    # Plot results
    plt.figure(figsize=(6,6))
    if (d0 == 1):
        plt.plot(Y0_test[0,:], upsilon, "o", markersize=2, label="Network")
        plt.plot(np.arange(start,end,0.01), f(np.arange(start,end,0.01)), label="Real")
        plt.xlabel("y")
        plt.ylabel("F(y)", rotation=90)
        plt.legend()
    elif (d0 == 2):
        ax = plt.axes(projection="3d")
        ax.plot(Y0_test[0,:], Y0_test[1,:], upsilon, "o", markersize=2, label="Network")
        X = np.arange(start[0], end[0], 0.1)
        Y = np.arange(start[1], end[1], 0.1)
        X, Y = np.meshgrid(X,Y)
        ax.plot_surface(X, Y, f(np.array([X,Y])), alpha=0.5)
        ax.set_xlabel(r'$y_1$')
        ax.set_ylabel(r'$y_2$')
        ax.set_zlabel(r'$F(y_1,y_2)$')
        ax.legend()
        ax.set_zlim([np.min(upsilon), np.max(upsilon)])
    
    # Convergence plot
    plt.figure(figsize=(6,6))
    y = resnet_dict["Js"][20:]
    x = resnet_dict["iter"][20:]
    plt.plot(x,y)
    plt.ylabel(r'$J(\theta)$', rotation=0)
    plt.xlabel("iterations")

    plt.show()