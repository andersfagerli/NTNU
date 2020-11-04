import numpy as np
import time

class ResNet:
    def __init__(self, K, d, tau, h, beta1 = 0.9, beta2 = 0.999, alpha = 0.01, eps = 1e-8):
        self.K = K      # Hidden layers
        self.d = d      # Dimension of hidden layers
        self.tau = tau  # Learning parameter for gradient descent
        self.h = h      # Layer weight

        self.activation_function = "sigmoid"
        self.hypothesis_function = "identity"

        # Initialize random weights
        self.W_shape = (K,d,d)
        self.W = np.random.random_sample(self.W_shape)
        self.b_shape = (K,d,1)
        self.b = np.random.random_sample(self.b_shape)
        
        self.w_shape = (d,1)
        self.w = np.random.random_sample(self.w_shape)
        self.mu_shape = (1,)
        self.mu = np.random.random_sample(self.mu_shape)

        # ADAM parameters
        self.v = np.zeros((np.prod(self.W_shape)+np.prod(self.b_shape)+np.prod(self.w_shape)+np.prod(self.mu_shape)))
        self.m = np.zeros((np.prod(self.W_shape)+np.prod(self.b_shape)+np.prod(self.w_shape)+np.prod(self.mu_shape)))

        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.eps = eps

    ### Activation functions ###
    def sigmoid(self, x):
        return np.tanh(x)
    
    def dsigmoid(self, x):
        return 1 / (np.cosh(x)**2)
    
    def ReLU(self, x):
        return np.maximum(x,0,x)
    
    def dReLU(self, x):
        dx = np.empty_like(x)
        dx[x < 0] = 0
        dx[x >= 0] = 1
        return dx

    def sigma(self, x):
        if self.activation_function == "relu":
            return self.ReLU(x)
        elif self.activation_function == "sigmoid":
            return self.sigmoid(x)
        else:
            print("ERROR: Unknown activation function, exiting ..")
            exit(1)
    
    def dsigma(self, x):
        if self.activation_function == "relu":
            return self.dReLU(x)
        elif self.activation_function == "sigmoid":
            return self.dsigmoid(x)
        else:
            print("ERROR: Unknown activation function, exiting ..")
            exit(1)
    
    def setActivationFunction(self, f):
        """
        Input:
            f : name of activation function as string ("relu", "sigmoid")
        """
        self.activation_function = f
    
    ### Hypothesis functions ###
    def identity(self, x):
        return x
    
    def didentity(self, x):
        rows = len(x)
        return np.ones((rows,1))
    
    def scaledTanh(self, x):
        return 0.5 * (1 + np.tanh(x/2))
    
    def dscaledTanh(self, x):
        return 0.5 / (np.cosh(x)+1)

    def eta(self, x):
        if self.hypothesis_function == "identity":
            return self.identity(x)
        elif self.hypothesis_function == "scaledTanh":
            return self.scaledTanh(x)
        else:
            print("ERROR: Unknown hypothesis function, exiting ..")
            exit(1)
    
    def deta(self, x):
        if self.hypothesis_function == "identity":
            return self.didentity(x)
        elif self.hypothesis_function == "scaledTanh":
            return self.dscaledTanh(x)
        else:
            print("ERROR: Unknown hypothesis function, exiting ..")
            exit(1)

    def setHypothesisFunction(self, f):
        """
        Input:
            f : name of hypothesis function as string ("identity", "scaledTanh")
        """
        self.hypothesis_function = f
    
    ### Transformation between layers ###
    def phi(self, Z, W, b):
        """
        Input:
            Z : (d x I) matrix of intermediate values from previous layer
            W : (d x d) matrix of parameter weights from previous layer
            b : (d x I) matrix of parameter weights, of form b=[b1 b2 .. bI] where each column is identical
        Output:
            phi : (d x I) matrix of intermediate values for next layer
        """
        return Z + self.h * self.sigma(W @ Z + b)    # eq.(4)

    def upsilon(self, Z_K, w, mu):
        """
        Input:
            Z_K : (d x I) matrix of intermediate values from last layer, K
            w : (d x 1) matrix of weights
            mu : scalar weight
        Output:
            upsilon : (I x 1) matrix of output values from the final layer
        """
        I = Z_K.shape[1]

        return self.eta(Z_K.T @ w + mu * np.ones((I,1)))    # eq.(5)
    
    ### Gradients ###
    def PK(self, c, Z_K, w, mu):
        """
        Input:
            c : (I x 1) matrix of data values
            Z_K : (d x I) matrix of intermediate values from last layer, K
            w : (d x 1) matrix of weights
            mu : scalar weight
        Output:
            PK : (d x I) matrix of gradient for final layer
        """
        I = Z_K.shape[1]

        upsilon = self.upsilon(Z_K, w, mu)
        deta = self.deta(Z_K.T @ w + mu * np.ones((I,1)))
        return np.outer(w, np.multiply(upsilon - c, deta))  # eq.(10)
    
    def Pprev(self, P, Z_prev, W_prev, b_prev):
        """
        Input:
            P : (d x I) matrix of gradient from next layer, P_{k}
            Z_prev : (d x I) matrix of intermediate values from previous layer, Z_{k-1}
            W_prev : (d x d) matrix of parameter weights from previous layer, W_{k-1}
            b_prev : (d x I) matrix of parameter weights, of form b=[b1 b2 .. bI] where each column is identical, from previous layer, b_{k-1}
        Output:
            Pprev : (d x I) matrix of gradient for previous layer
        """
        dsigma = self.dsigma(W_prev @ Z_prev + b_prev)
        return P + self.h * W_prev.T @ np.multiply(dsigma, P)   # eq.(11)
    
    def dJdmu(self, c, Z_K, w, mu):
        """
        Input:
            c : (I x 1) matrix of data values
            Z_K : (d x I) matrix of intermediate values from last layer, K
            w : (d x 1) matrix of weights
            mu : scalar weight
        Output:
            dJdmu : scalar gradient dJ/dmu
        """
        I = Z_K.shape[1]

        deta = self.deta(Z_K.T @ w + mu * np.ones((I,1)))
        upsilon = self.upsilon(Z_K, w, mu)
        return deta.T @ (upsilon - c)   # eq.(8)
    
    def dJdw(self, c, Z_K, w, mu):
        """
        Input:
            c : (I x 1) matrix of data values
            Z_K : (d x I) matrix of intermediate values from last layer, K
            w : (d x 1) matrix of weights
            mu : scalar weight
        Output:
            dJdw : (d x 1) matrix gradient dJ/dw
        """
        I = Z_K.shape[1]

        deta = self.deta(Z_K.T @ w + mu * np.ones((I,1)))
        upsilon = self.upsilon(Z_K, w, mu)
        return Z_K @ np.multiply(upsilon - c, deta) # eq.(9)
    
    def dJdWk(self, Zk, Wk, bk, P_next):
        """
        Input:
            Zk : (d x I) matrix of intermediate values from layer k
            Wk : (d x d) matrix of parameter weights from layer k
            bk : (d x I) matrix of parameter weights, of form b=[b1 b2 .. bI] where each column is identical, from layer k
            P_next : (d x I) matrix of gradient for next layer, P_{k+1}
        Output:
            dJdWk : (d x d) matrix gradient dJ/dWk
        """
        dsigma = self.dsigma(Wk @ Zk + bk)
        return self.h * np.multiply(P_next, dsigma) @ Zk.T  # eq.(12)

    def dJdbk(self, Zk, Wk, bk, P_next):
        """
        Input:
            Zk : (d x I) matrix of intermediate values from layer k
            Wk : (d x d) matrix of parameter weights from layer k
            bk : (d x I) matrix of parameter weights, of form b=[b1 b2 .. bI] where each column is identical, from layer k
            P_next : (d x I) matrix of gradient for next layer, P_{k+1}
        Output:
            dJdbk : (d x 1) matrix gradient dJ/dbk
        """
        I = Zk.shape[1]
        
        dsigma = self.dsigma(Wk @ Zk + bk)
        return self.h * np.multiply(P_next, dsigma) @ np.ones((I,1))    # eq.(13)
    
    ### Objective function ###
    def J(self, c, Z_K, w, mu):
        """
        Input:
            c : (I x 1) matrix of output data
            Z_K : (d x I) matrix of intermediate values from last layer, K
            w : (d x 1) matrix of weights
            mu : scalar weight
        Output:
            J : scalar value of objective function
        """
        upsilon = self.upsilon(Z_K, w, mu)
        return 0.5 * np.linalg.norm(upsilon - c)**2 # eq.(6)
    
    ### Optimization ###
    def gradientDescent(self, thetas, dthetas, m=None, v=None):
        """
        Input:
            thetas : numpy array of parameters, e.g np.array([W, b, w, mu])
            dthetas : numpy array of parameter gradients, e.g np.array([dJ/dW, dJ/db, dJ/dw, dJ/dmu])
        Output:
            thetas_upd : updated array of parameters, e.g np.array([Wupd, bupd, wupd, muupd])
        """

        return thetas - self.tau * dthetas

    def adamsDescent(
        self,
        thetas,
        dthetas,
        beta1 = 0.9,
        beta2 = 0.999,
        alpha = 0.01,
        eps = 1e-8
    ):

        g = dthetas
        self.m = beta1*self.m + (1-beta1)*g
        self.v = beta2*self.v + (1-beta2)*g**2
        m_hat = self.m / (1 - beta1)
        v_hat = self.v / (1 - beta2)
        thetas = thetas - alpha*m_hat/(np.sqrt(v_hat) + eps)

        thetas = np.array([
            thetas[:np.prod(self.W_shape)].reshape(self.W_shape),
            thetas[np.prod(self.W_shape):(np.prod(self.W_shape)+np.prod(self.b_shape))].reshape(self.b_shape),
            thetas[(np.prod(self.W_shape)+np.prod(self.b_shape)):(np.prod(self.W_shape)+np.prod(self.b_shape)+np.prod(self.w_shape))].reshape(self.w_shape),
            thetas[(np.prod(self.W_shape)+np.prod(self.b_shape)+np.prod(self.w_shape)):(np.prod(self.W_shape)+np.prod(self.b_shape)+np.prod(self.w_shape)+np.prod(self.mu_shape))].reshape(self.mu_shape)
        ])

        return thetas

    ### Scaling of data ###
    def scale(self, x, alpha=0, beta=1):
        """
        Input:
            x : data to be scaled, must be a numpy array or a scalar
            alpha : lower bound on scaled data
            beta : upper bound on scaled data
        Output:
            x_scaled : scaled data
            a : min(x), minimum value of x
            b : max(x), maximum value of x
        """
        a = np.min(x)
        b = np.max(x)

        x_scaled = 1/(b-a) * ((b-x)*alpha + (x-a)*beta)
        return x_scaled, a, b

    def scale_gradient(self, x, alpha=0, beta=1):
        """
        Input:
            x : data to be scaled, must be a numpy array or a scalar
            alpha : lower bound on scaled data
            beta : upper bound on scaled data
        Output:
            x_scaled : scaled data
            a : min(x), minimum value of x
            b : max(x), maximum value of x
        """
        a = np.min(x)
        b = np.max(x)

        x_scaled = (beta - alpha)/(b-a) * x
        return x_scaled, a, b
    
    def rescale(self, x_scaled, a, b, alpha=0, beta=1):
        """
        Input:
            x_scaled : data to be rescaled, must be a numpy array or a scalar
            a : min(x), minimum of original data
            b : max(x), maximum of original data
            alpha : lower bound on scaled data
            beta : upper bound on scaled data
        Output:
            x : rescaled data
        """
        x = 1/(beta-alpha) * ((beta-x_scaled)*a + (x_scaled - alpha)*b)
        return x

    def rescale_gradient(self, x_scaled, a, b, alpha=0, beta=1):
        """
        Input:
            x_scaled : data to be rescaled, must be a numpy array or a scalar
            a : min(x), minimum of original data
            b : max(x), maximum of original data
            alpha : lower bound on scaled data
            beta : upper bound on scaled data
        Output:
            x : rescaled data
        """
        x = 1/(beta-alpha) * ((-x_scaled)*a + (x_scaled)*b)
        return x
    
    ### Check for convergence ###
    def hasConverged(self, J_current, J_prev, J_threshold, residual_threshold):
        """
        Input:
            J_current : objective function value from current iteration
            J_prev : objective function value from previous iteration
            J_threshold : threshold for value of objective function
            residual_threshold : threshold for residual
        Output:
            converged : True/False
        """
        if np.abs(J_current - J_prev) < residual_threshold and J_current < J_threshold:
            return True
        else:
            return False

    ### Main algorithm ###
    def train(self, Y0, c, J_threshold=0.05, residual_threshold=0.001, max_iterations=1e5, data_output=False, debug=False):
        """
        Input:
            Y0 : (d0 x I) matrix of input training data
            c : (I x 1) matrix of output training data
            J_threshold : threshold for value of objective function
            residual_threshold : threshold for residual
            max_iterations: maximum number of iterations before terminating
            data_output : flag for outputting data
            debug : flag for debug printouts
        Output:
            Js : if data_output=True, return relevant data
        """
        d0, I = Y0.shape

        assert(d0 <= self.d)
        
        Z = np.zeros((self.K + 1, self.d, I))           # [Z0, Z1, ... , ZK]
        P = np.zeros((self.K + 1, self.d, I))           # [P0, P1, ... , PK]
        dJdW = np.zeros((self.K, self.d, self.d))       # [dJ/dW0, dJ/dW1, ... , dJ/dWK-1]
        dJdb = np.zeros((self.K, self.d, 1))            # [dJ/db0, dJ/db1, ... , dJ/dbK-1]
        Js = np.zeros(int(max_iterations+1))            # Store value of objective function in each iteration for visualization

        Z[0,:d0,:] = Y0
        Js[0] = self.J(c, Z[self.K,:,:], self.w, self.mu)
        
        converged = False
        iteration = 0
        start = time.time()
        while (iteration < max_iterations and not converged):
            for k in range(self.K):
                Z[k+1,:,:] = self.phi(Z[k,:,:], self.W[k,:,:], self.b[k,:,:])
            
            P[self.K,:,:] = self.PK(c, Z[self.K,:,:], self.w, self.mu)

            dJdmu = self.dJdmu(c, Z[self.K,:,:], self.w, self.mu)
            dJdw = self.dJdw(c, Z[self.K,:,:], self.w, self.mu)

            for k in range(self.K, 1, -1):
                P[k-1,:,:] = self.Pprev(P[k,:,:], Z[k-1,:,:], self.W[k-1,:,:], self.b[k-1,:,:])
            
            for k in range(self.K):
                dJdW[k,:,:] = self.dJdWk(Z[k,:,:], self.W[k,:,:], self.b[k,:,:], P[k+1,:,:])
                dJdb[k,:,:] = self.dJdbk(Z[k,:,:], self.W[k,:,:], self.b[k,:,:], P[k+1,:,:])

            # Optimize
            thetas = np.array([self.W, self.b, self.w, self.mu])
            dthetas = np.array([dJdW, dJdb, dJdw, dJdmu])

            ## ADAM
            dthetas = np.hstack([r.ravel() for r in dthetas]).ravel()
            thetas = np.hstack([r.ravel() for r in thetas]).ravel()

            self.W, self.b, self.w, self.mu = self.adamsDescent(
                thetas, dthetas,
                beta1=self.beta1,
                beta2=self.beta2,
                alpha=self.alpha,
                eps=self.eps
            )
            
            ### Uncomment to use gradient descent instead
            # self.W, self.b, self.w, self.mu = self.gradientDescent(thetas, dthetas)

            # Check convergence
            Js[iteration+1] = self.J(c, Z[self.K,:,:], self.w, self.mu)
            if self.hasConverged(Js[iteration+1], Js[iteration], J_threshold, residual_threshold):
                converged = True

            if (debug is True):
                if (iteration % 50 == 0):
                    print(f"Iteration: {iteration}\ttime: {time.time() - start} s")
                    print("J: ", Js[iteration+1])
                    print("")

            iteration += 1

        if (data_output is True):
            Js = Js[Js != 0]
            comp_time = time.time()-start
            return {"Js":Js, "iter":np.arange(iteration+1),"comp_time":comp_time}
    
    ### Run data through trained network ###
    def F(self, Y0):
        """
        Input:
            Y0 : (d0 x I) matrix of input test data
        Output:
            upsilon : (1 x I) vector of output test data
        """
        Z = self.Zfwdsweep(Y0)

        return self.upsilon(Z[self.K,:,:], self.w, self.mu)[:,0]

    def Zfwdsweep(self, Y0):
        if len(Y0.shape) == 1:
            d0 = 1
            I = Y0.shape[0]
        else:
            d0, I = Y0.shape 

        Z = np.zeros((self.K + 1, self.d, I))
        Z[0,:d0,:] = Y0

        assert(d0 <= self.d)

        for k in range(self.K):
            Z[k+1,:,:] = self.phi(Z[k,:,:], self.W[k,:,:], self.b[k,:,:])

        return Z

    ### Functions for computing gradient, nabla(F)
    def dG(self, y, w, mu):
        return self.deta(w.T@y + mu)*w

    def dphi_transpose_A(self, A, W, Z, b):
        assert len(b.shape) == 2 and b.shape[1] == 1 and b.shape[0] == W.shape[0], "b wrong shape"
        return A + W.T@(self.h*self.dsigma(W@Z + b)*A)

    def gradient(self, y):

        z = self.Zfwdsweep(y)

        A = self.dG(z[self.K,:,:], self.w, self.mu)
        for k in range(self.K, 0, -1):
            A = self.dphi_transpose_A(A, self.W[k-1,:,:], z[k-1,:,:], self.b[k-1,:,:])

        if len(y.shape) == 1:
            d0 = 1
        else:
            d0 = y.shape[0]

        A = A[:d0]
        if A.shape[0] == 1:
            A = A.ravel()

        return A

