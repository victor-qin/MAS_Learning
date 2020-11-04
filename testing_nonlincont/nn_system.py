import numpy as np
import scipy.integrate as integrate

'''##########################################'''
''' NONLINEAR SYSTEM FOR TESTING - INVERTED PENDULUM'''
'''##########################################'''
class CartPoleNNCont:

    def __init__(self, M = 0.5, m = 0.2, c = 0.1, gamma = 0.1, I = 0.006, l=0.3, 
                    t_bound = 10, alpha = 0.001, radius = 0.02, num = 3):
        self.g = 9.81
        
        # mass of cart
        self.M = M

        # mass of bar
        self.m = m

        # inertia of cart
        self.I = I

        # frictive coefficient
        self.c = c
        self.gamma = gamma

        # pendulum length
        self.l = l

        # time variables
        self._res = 0.01
        self._t_bound = t_bound
        self.times = np.linspace(0, t_bound, num = int(t_bound / self._res) + 1)

        self.alpha = alpha
        self.radius = radius

        self.num = 1

        # temporary variables for smuggling across ODE functions
        self.inputs_temp = None

    # define linear control policy - update when you want to change it
    # takes in 2-vector, outputs a single control
    # assume they're numpy matrices
    def setLinearControl(self, Kp, gain, x0, target = None):

        self.Kp = Kp    # 1 x 2 matrix
        self.gain = gain    # scalar
        self.x0 = x0        # starting point
        self.target = target        # the target

        if(target is None):
            self.target = np.zeros(self.x0.shape)


        self.Kp_test = self.Kp
        self.gain_test = self.gain
        

    # define the cost function, position, input, final errors
    def setCost(self, Q, R, F = None, gamma=None):

        self.Q = Q
        self.R = R

        self.F = None
        self.gammas = None

        if(F is not None):
            self.F = F

        if(gamma is not None):
            self.gammas = np.power(gamma, self.times)


    # define the dynamics based on timestamps, current control policy, initial value
    def dynamics(self, y_all, t):
        # u = (Kp @ y + gain)
        # u = (Kp @ y  + gain)

        y = y_all[:4]

        M_t = self.M + self.m
        J_t = self.I + self.m * self.l**2
        
        u = self.Kp_test @ (y[:4] - self.target[:4]) + self.gain_test
        # u = 0
        # if(self.inputs_temp is not None):
        #     self.inputs_temp.append(u)

        # y = 
        # y_0 = [y[2]]
        # y_1 = [y[3]]

        y_2 = ((-(self.m * self.l * np.sin(y[1]) * y[3]**2)
                + self.m * self.g * (self.m * self.l**2 / J_t) * np.sin(y[1]) * np.cos(y[1])
                - self.c * y[2] - self.gamma * self.l * self.m * np.cos(y[1]) * y[3]
                + u ) / 
                (M_t - self.m * (self.m * self.l**2 / J_t) * np.cos(y[1])**2)
            )
        
        y_3 = ((-(self.m * self.l * np.sin(y[1]) * np.cos(y[1]) * y[3]**2)
                + M_t * self.g * self.l * np.sin(y[1])
                - self.c * self.l * np.cos(y[1]) * y[2]
                - self.gamma * (M_t / self.m) * y[3] + self.l * np.cos(y[1]) * u) / 
                (J_t * (M_t / self.m) - self.m * (self.l * np.cos(y[1]))**2
                )
            )

        # ydot = self.A @ y + self.B @ u
        # ydot = np.array([y_0, y_1, y_2, y_3])
        ydot = np.concatenate(([y[2]], [y[3]], y_2, y_3, u))

        return ydot

    # calculates cost function for updating
    def calcCost(self, error, input):
        
        # basic cost matrix calc
        cost = (error * (self.Q @ error.T).T + input * (self.R.T @ input.T).T)
        # cost = 

        # if time discounting
        if(self.gammas is not None):
            cost = np.dot(self.gammas, cost)

        # if there's a final cost
        if(self.F is not None):
            return np.sum(cost / error.shape[0]) + error[-1, :] @ self.F @ error[-1, :].T

        # return the cost averaged
        return np.sum(cost / error.shape[0])

    # runs the simulation for an iteration - takes in an initial position and set of times
    # returns output and the total error for the system
    def runSimulation(self, x0):
        # self.inputs_temp = []
        output = integrate.odeint(
            func = self.dynamics,
            y0 = x0,
            t = self.times
        )

        states = output[:, :4]
        inputs = output[:, 4:]

        cost = self.calcCost(np.abs(states - self.target[:4]), np.array(inputs))
        return states, cost

    # runs an epoch of N iterations, which then gives the error at the end
    # update controller between each iteration
    # used for running the epoch in multiagent, which gets pulled out to be average
    def runEpoch(self, N):
        for i in range(N):

             # pick a vector on unit sphere to move Kp in
            # vec = (np.random.rand(self.Kp.size + self.gain.size) - 0.5)
            vec = (np.random.rand(self.Kp.size + 1) - 0.5)
            vec = self.radius * vec / np.linalg.norm(vec)
            vec_k = np.reshape(vec[:self.Kp.size], self.Kp.shape)
            vec_k[0] = 0
            vec_k[2] = 0
            vec_g = vec[-1:]


            rand_start = self.x0#  + np.random.randn(self.x0.size) * 0.1
            # _, err1 = self.runSimulation(self.Kp + vec_k, self.gain + vec_g, rand_start)
            # two-point gradient descent estimate
            self.Kp_test = self.Kp + vec_k
            self.gain_test = self.gain + vec_g
            _, err1 = self.runSimulation(rand_start)

            self.Kp_test = self.Kp - vec_k
            self.gain_test = self.gain - vec_g
            _, err2 = self.runSimulation(rand_start)

            print("Trial %d: error is %f" % (i, err2))

            # plt.plot(system.times, output)
            # plt.show()

            # take a step
            z_step =  (self.Kp.size + self.gain.size / (2 * self.radius)) * (err1 - err2)
            self.Kp = self.Kp - self.alpha * z_step * vec_k
            self.gain = self.gain - self.alpha / self.num * z_step * vec_g


        # grab error for return
        self.Kp_test = self.Kp
        self.gain_test = self.gain
        _, error = self.runSimulation(rand_start)

        return error

'''##########################################'''
''' LINEAR SYSTEM FOR TESTING'''
'''##########################################'''
class LinSystemNNCont:

    def __init__(self, t_bound = 10, alpha = 0.001, radius = 0.02, num = 3):

        self.A = np.array([[1,   0,  -10],
                        [-1,  1,  0],
                        [0,   0,  1]])

        self.B = np.array([[1,   -10,    0],
                        [0,   1,  0],
                        [-1,  0,  1]])

        

        # time variables
        self._res = 0.01
        self._t_bound = t_bound
        self.times = np.linspace(0, t_bound, num = int(t_bound / self._res) + 1)

        self.alpha = alpha
        self.radius = radius

        self.num = 1

        # temporary variables for smuggling across ODE functions
        self.inputs_temp = None

    # define linear control policy - update when you want to change it
    # takes in 2-vector, outputs a single control
    # assume they're numpy matrices
    def setLinearControl(self, Kp, gain, x0, target = None):

        self.Kp = Kp    # 1 x 2 matrix
        self.gain = gain    # scalar
        self.x0 = x0        # starting point
        self.target = target        # the target

        if(target is None):
            self.target = np.zeros(self.x0.shape)

        self.Kp_test = self.Kp
        self.gain_test = self.gain

        # raise NotImplementedError

    # define the cost function, position, input, final errors
    def setCost(self, Q, R, F = None, gamma=None):

        self.Q = Q
        self.R = R

        self.F = None
        self.gammas = None

        if(F is not None):
            self.F = F

        if(gamma is not None):
            self.gammas = np.power(gamma, self.times)


    # define the dynamics based on timestamps, current control policy, initial value
    def dynamics(self, y_all, t):


        y = y_all[:3]        
        u = self.Kp_test @ y + self.gain_test
        # u = 0

        y_change = self.A @ y + self.B @ u
        # ydot = np.array([y_0, y_1, y_2, y_3])
        ydot = np.concatenate((y_change, u))

        return ydot

    # calculates cost function for updating
    def calcCost(self, error, input):
        
        # basic cost matrix calc
        cost = (error * (self.Q @ error.T).T + input * (self.R.T @ input.T).T)

        # if time discounting
        if(self.gammas is not None):
            cost = np.dot(self.gammas, cost)

        # if there's a final cost
        if(self.F is not None):
            return np.sum(cost / error.shape[0]) + error[-1, :] @ self.F @ error[-1, :].T

        # return the cost averaged
        return np.sum(cost / error.shape[0])

    # runs the simulation for an iteration - takes in an initial position and set of times
    # returns output and the total error for the system
    def runSimulation(self, x0):
        # self.inputs_temp = []
        output = integrate.odeint(
            func = self.dynamics,
            y0 = x0,
            t = self.times
        )

        states = output[:, :3]
        inputs = output[:, 3:]

        cost = self.calcCost(np.abs(states), np.array(inputs))
        return states, cost

    # runs an epoch of N iterations, which then gives the error at the end
    # update controller between each iteration
    # used for running the epoch in multiagent, which gets pulled out to be average
    def runEpoch(self, N):
        for i in range(N):

             # pick a vector on unit sphere to move Kp in
            # vec = (np.random.rand(self.Kp.size + self.gain.size) - 0.5)
            vec = (np.random.rand(self.Kp.size + self.gain.size) - 0.5)
            vec = self.radius * vec / np.linalg.norm(vec)
            vec_k = np.reshape(vec[:self.Kp.size], self.Kp.shape)
            vec_g = vec[self.Kp.size:]


            rand_start = self.x0#  + np.random.randn(self.x0.size) * 0.1
            # _, err1 = self.runSimulation(self.Kp + vec_k, self.gain + vec_g, rand_start)
            # two-point gradient descent estimate
            self.Kp_test = self.Kp + vec_k
            self.gain_test = self.gain + vec_g
            _, err1 = self.runSimulation(rand_start)

            self.Kp_test = self.Kp - vec_k
            self.gain_test = self.gain - vec_g
            _, err2 = self.runSimulation(rand_start)

            print("Trial %d: error is %f" % (i, err2))

            # plt.plot(system.times, output)
            # plt.show()

            # take a step
            z_step =  (self.Kp.size + self.gain.size / (2 * self.radius)) * (err1 - err2)
            self.Kp = self.Kp - self.alpha * z_step * vec_k
            self.gain = self.gain - self.alpha / self.num * z_step * vec_g


        # grab error for return
        self.Kp_test = self.Kp
        self.gain_test = self.gain
        _, error = self.runSimulation(rand_start)

        return error