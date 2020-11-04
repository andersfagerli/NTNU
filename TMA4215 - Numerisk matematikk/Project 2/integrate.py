import numpy as np

def integrate(p0, q0, h: float, Tgrad, Vgrad, T: float, int_step):
    """
    Numerical integration
    Input:

        p0: Initial state p, dim: (np,)
        q0: Initial state q  dim: (nq,)
        h: Time step length
        Tgrad: Learnt gradient of T, del T / del p
        Vgrad: Learnt gradient of V, del V / del q
        T: Final time of integration
        int_step: The integration step is done, symplectic or Størmer-Verlet
    Output:
        p: Solution of p over all time steps, dim: (N, np)
        q: Solution of q over all time steps  dim: (N, nq)
    """

    N = int(np.ceil(T/h))

    try:
        p = np.zeros((N, *p0.shape))
        q = np.zeros((N, *q0.shape))
    except AttributeError:
        p = np.zeros(N)
        q = np.zeros(N)

    p[0] = p0
    q[0] = q0

    for n in range(N - 1):
        q[n + 1], p[n + 1] = int_step(q[n], p[n], h, Tgrad, Vgrad)

    return p, q

def symplectic_euler(pn: np.array, qn: np.array, h: float, Tgrad, Vgrad):
    """
    Symplectic Euler for integration
    Input:
        pn: State p in time step n, dim: (np,)
        qn: State q in time step n,  dim: (nq,)
        h: Time step length
        Tgrad: Learnt gradient of T, del T / del p
        Vgrad: Learnt gradient of V, del V / del q
    Output:
        p: Integrated p, p in time step n + 1, dim: (np,)
        q: Integrated q, q in time step n + 1, dim: (nq,)
    """

    q = qn + h*Tgrad(pn)
    p = pn - h*Vgrad(q)

    return p, q

def stormer_verlet(pn: np.array, qn: np.array, h: float, Tgrad, Vgrad):
    """
    Størmer-Verlet method for integration of Hamiltonian systems
    Input:
        pn: State p in time step n, dim: (np,)
        qn: State q in time step n,  dim: (nq,)
        h: Time step length
        Tgrad: Learnt gradient of T, del T / del p
        Vgrad: Learnt gradient of V, del V / del q
    Output:
        p: Integrated p, p in time step n + 1, dim: (np,)
        q: Integrated q, q in time step n + 1, dim: (nq,)
    """

    p = pn - h/2*Vgrad(qn)
    q = qn + h*Tgrad(p)
    p = p - h/2*Vgrad(q)

    return p, q


if __name__ == "__main__":
    import hamiltonians as hs
    import matplotlib.pyplot as plt

    m = 10
    g = 9.81
    l = 1

    pendulum = hs.NonlinearPendulum(m, g, l)

    Vgrad = pendulum.Vgrad()
    Tgrad = pendulum.Tgrad()

    p0 = 0
    q0 = np.pi / 2

    h = 0.01
    T = 10

    p, q = integrate(p0, q0, h, Tgrad, Vgrad, T, symplectic_euler)

    t = np.linspace(0, T, len(p))

    plt.plot(t, p, label='p')
    plt.plot(t, q, label='q')

    plt.legend()
    plt.show()

    