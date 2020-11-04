import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


# Goal: simulate a linear system, eventually add in control inputs, then add in grad descent on the control inputs
# A is R^3x3, x0 is R3, max_t is maximum simulation time
class LinearSystem(object):

    def __init__(self, A, B, x0, t_bound, Kp = None, x_targ = None, g = None):

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


        self._res = 0.01
        # define the function for RK45
        def function(y, t):
            ydot = A @ y + B @ Kp @ y + B @ gain
            return(ydot)

        # define the differential system
        # self.system = integrate.RK45(fun = function,
        #     t0 = 0.0,
        #     y0 = x0,
        #     t_bound = t_bound,
        #     first_step = 0.1,
        #     max_step = 0.1)
        self._t_bound = t_bound
        self.times = np.linspace(0, t_bound, num = int(t_bound / self._res) + 1)
        self.system = integrate.odeint(
            func = function,
            y0 = x0,
            t = self.times
        )

        self.time = []
        self.output = []

    # run the full simulation
    def simulate(self):
        
        x = self.x0
        output = np.zeros((int(self._t_bound / self._res) + 1, self.x0.size))
        output[0, :] = x
        for i in range(int(self._t_bound / self._res)):
            xdot = self.A @ x + self.B @ self.Kp @ x + self.B @ self.gain

            x = x + xdot * self._res
            output[i + 1, :] = x

        return(output)

        # self.time.append(self.system.t)
        # self.output.append(self.system.y)

        # # run the system until it runs
        # while(self.system.status == "running"):
        #     self.system.step()
        #     self.time.append(self.system.t)
        #     self.output.append(self.system.y)


if __name__ == "__main__":

    A = np.array([[1,   0,  10],
                  [-1,  1,  0],
                  [0,   0,  1]])

    B = np.array([[1,   -10,    0],
                  [0,   1,  0],
                  [-1,  0,  1]])

    Kp =  -1 * np.array([[3,  0,  0], 
                         [0,   3,  0],
                         [0,   0,  3]])

    target = np.array([3, 4, 5])
    gain = np.linalg.inv(B) @ (-(A + B @ Kp) @ target)


    print(np.linalg.eigvals(A + np.matmul(B, Kp)))
    print(np.linalg.eigvals(A + np.matmul(B, Kp) + B @ gain))
    print(-np.linalg.inv((A + B @ Kp)) @ (B @ gain))

    system = LinearSystem(A, B, np.array([4, 4, 4]), 10, Kp = Kp, x_targ = target, g = gain)
    output = system.simulate()
    # print(system.system)
    plt.plot(system.times, system.system)

    plt.plot(system.times, output)
    plt.show()
    # print(system.time)
    # print(out)