import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Goal: simulate a linear system, eventually add in control inputs, then add in grad descent on the control inputs
# A is R^3x3, x0 is R3, max_t is maximum simulation time
class LinearSystem(object):

    def __init__(self, A, B, x0, t_bound, Kp = None, x_targ = None, g = None, gamma=0.9, N = 1):

        # check if we're using gain
        if(g.any() == None):
            gain = np.zeros(x0.size)
        else:
            gain = g

        # system variables that don't change between runs
        self.A = A
        self.B = B
        self.t_bound = t_bound

        # testing variables that do change
        self.Kp = Kp
        self.gain = gain

        self.x0 = x0
        self.x_targ = x_targ

        self._res = 0.01
        self._t_bound = t_bound
        self.times = np.linspace(0, t_bound, num = int(t_bound / self._res) + 1)
        self.gammas = np.power(gamma, self.times)

        self.N = N

    # error is the entire output - target matrix
    def cost_J(self, error, input):

        Q = np.array([[2, -1, 0],
                      [-1, 2, -1],
                      [0, -1, 2]])

        R = np.array([[5, -3, 0],   
                      [-3, 5, -2],
                      [0, -2, 5]])

        # F = np.array([[6, 0, 0],
                    #   [0, 6, 0],
                    #   [0, 0, 6]])

        cost = np.dot(self.gammas, (error * (Q @ error.T).T + input.T * (R @ input).T))
        return np.sum(cost / error.shape[0]) #+ error[-1, :] @ F @ error[-1, :].T

    # run the full simulation
    def simulate(self, Kp, gain, x0):

        # define the function for RK45
        # self.count = 0
        def function(y, t):
            # u = (Kp @ y + gain)
            u = (Kp @ y  + gain)

            ydot = self.A @ y + self.B @ u

            # print(np.concatenate((ydot, u)))
            return ydot

        output = integrate.odeint(
            func = function,
            y0 = x0,
            t = self.times
        )
        # print(output)
        # x = x0
        # output = np.zeros((int(self._t_bound / self._res) + 1, self.x0.size))
        # output[0, :] = x
        # for i in range(int(self._t_bound / self._res)):
        #     xdot = self.A @ x + self.B @ self.Kp @ x + self.B @ self.gain

        #     x = x + xdot * self._res
        #     output[i + 1, :] = x

        error = self.cost_J(np.abs(output - self.x_targ), Kp @ output.T + np.reshape(np.tile(gain, output.shape[0]), output.shape).T)
        # error = self._res * np.sum(np.linalg.norm(output - self.x_targ, axis=1))
        return output, error

    def zero_descent(self, alpha = 0.02, epsilon = 0.5, iter = 1):

        radius = 0.01 

        for i in range(iter):
            # pick a vector on unit sphere to move Kp in
            vec = (np.random.rand(self.Kp.size + self.gain.size) - 0.5)
            vec = radius * vec / np.linalg.norm(vec)
            vec_k = np.reshape(vec[:self.Kp.size], self.Kp.shape)
            vec_g = vec[-self.gain.size:]


            rand_start = self.x0#  + np.random.randn(self.x0.size) * 0.1
            # two-point gradient descent estimate
            _, err1 = self.simulate(self.Kp + vec_k, self.gain + vec_g, rand_start)
            output, err2 = self.simulate(self.Kp - vec_k, self.gain - vec_g, rand_start)

            print("Trial %d: error is %f" % (i, err2))

            # plt.plot(system.times, output)
            # plt.show()

            # take a step
            z_step =  (self.Kp.size + self.gain.size / (2 * radius)) * (err1 - err2)
            self.Kp = self.Kp - alpha * z_step * vec_k
            self.gain = self.gain - alpha / self.N * z_step * vec_g

        return (output, err2)


if __name__ == "__main__":

    A = np.array([[1,   0,  -10],
                  [-1,  1,  0],
                  [0,   0,  1]])

    B = np.array([[1,   -10,    0],
                  [0,   1,  0],
                  [-1,  0,  1]])

    Kp =  -1 * np.array([[3,  0,  0], 
                         [0,   3,  0],
                         [0,   0,  3]])

    target1 = np.array([0, 0, 0])
    target2 = np.array([-1, -1, -1])
    # gain = np.linalg.inv(B) @ (-(A + B @ Kp) @ target)
    # print(gain)
    gain = np.array([0, 0, 0])
    # gain = np.random.randn(3) * 4


    # print(np.linalg.eigvals(A + np.matmul(B, Kp)))
    # print(np.linalg.eigvals(A + np.matmul(B, Kp) + B @ gain))
    # print(-np.linalg.inv((A + B @ Kp)) @ (B @ gain))
    ''' Singe agent plot '''
    N = 1
    system1 = LinearSystem(A, B, np.array([4, 4, 4]), 10, Kp = Kp, x_targ = target1, g = gain, N = N)
    system2 = LinearSystem(A, B, np.array([4, 4, 4]), 10, Kp = Kp, x_targ = target2, g = gain, N = N)
    system3 = LinearSystem(A, B, np.array([4, 4, 4]), 10, Kp = Kp, x_targ = target2, g = gain, N = N)
    error = []
    for i in range(100):

        _, err1 = system1.zero_descent(alpha = 0.06)
        _, err2 = system2.zero_descent(alpha = 0.06)
        _, err3 = system3.zero_descent(alpha = 0.06)

        err = (err1 + err2 + err3) / 3
        error.append(err)

    # simulation plot if you want it
    # output, _ = system.simulate(system.Kp, system.gain, system.x0)
    # plt.plot(system.times, output)
    # plt.show()

    plt.figure(1)
    plt.plot(np.linspace(1, 100, 100), error, label="single")
    plt.legend()
    # plt.show()


    '''Plot for two agents'''
    N = 3
    system1 = LinearSystem(A, B, np.array([4, 4, 4]), 10, Kp = Kp, x_targ = target1, g = gain, N = N)
    system2 = LinearSystem(A, B, np.array([4, 4, 4]), 10, Kp = Kp, x_targ = target2, g = gain, N = N)
    system3 = LinearSystem(A, B, np.array([4, 4, 4]), 10, Kp = Kp, x_targ = target2, g = gain, N = N)
    error = []
    
    for i in range(100):

        _, err1 = system1.zero_descent(alpha = 0.06)
        _, err2 = system2.zero_descent(alpha = 0.06)
        _, err3 = system3.zero_descent(alpha = 0.06)

        err = (err1 + err2 + err3) / N

        K_avg = (system1.Kp + system2 .Kp + system3.Kp) / N
        # index = np.argmin([err1, err2, err3])
        # Ks = [system1.Kp, system2 .Kp, system3.Kp]
        # K_avg = Ks[index]
        system1.Kp = K_avg
        system2.Kp = K_avg

        error.append(err)
    
    # plt.figure(1)
    plt.plot(np.linspace(1, 100, 100), error, label="multi")
    plt.legend()
    plt.show()


    # output = system.zero_descent()
    # output, error = system.simulate()
    # print(system.system)

    '''Plot for system trials '''
    # plt.figure(2)
    # output, _ = system1.simulate(system1.Kp, system1.gain, system1.x0)
    # plt.plot(system1.times, output)
    # plt.show()

    # plt.figure(3)
    # output, _ = system2.simulate(system2.Kp, system2.gain, system2.x0)
    # plt.plot(system2.times, output)
    # plt.show()

    print('done')
    # plt.plot(system.times, output)
    # plt.show()

    # print(system.Kp, system.gain)
    # print(np.linalg.inv(B) @ (-(A + B @ system.Kp) @ target))

