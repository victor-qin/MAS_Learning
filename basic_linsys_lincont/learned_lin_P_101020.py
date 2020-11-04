import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp


# Goal: simulate a linear system, eventually add in control inputs, then add in grad descent on the control inputs
# A is R^3x3, x0 is R3, max_t is maximum simulation time
class LinearSystem(object):

    def __init__(self, A, B, x0, t_bound, Kp = None, x_targ = None, g = None):

        # # check if we're using gain
        # if(g.any() == None):
        #     gain = np.zeros(x0.size)
        # else:
        #     gain = g

        # system variables that don't change between runs
        self.A = A
        self.B = B
        self.t_bound = t_bound

        # testing variables that do change
        self.Kp = Kp
        self.gain = gain

        self.x0 = x0
        self.x_targ = x_targ
        self.times = tf.linspace(0, t_bound, num = int(t_bound / 0.01) + 1)

        # define things for the differential system later
        self.output = None


    # run the full simulation
    def simulate(self):


        # define the function for RK45
        def function(y, t):
            ydot = self.A @ y + self.B @ self.Kp.numpy() @ y + self.B @ self.gain.numpy()
            return(ydot)

        # define the differential system
        self.output = integrate.odeint(
            func = function,
            y0 = self.x0,
            t = self.times
        )

        # error = tf.reduce_sum(tf.linalg.normalize(self.output - self.x_targ, axis = 1)[1])
        # print(error)
        # return(error)

    def tuning(self, learning_rate):

        Kprop = self.Kp
        gainer = self.gain

        @tf.function
        def add(a,b,c):
            return tf.constant(a + b + c, dtype=tf.float64)

        @tf.function(input_signature=(
            tf.TensorSpec(shape=None, dtype=tf.float64),
            tf.TensorSpec(shape=(3,), dtype=tf.float64),
            tf.TensorSpec(shape=(3,3), dtype=tf.float64),
            tf.TensorSpec(shape=(3,), dtype=tf.float64)))
        def func(t, y, K, g):
            tf.print("internal")
            # y = tf.convert_to_tensor(y, dtype=tf.float64)
            return add(tf.linalg.matvec(self.A, y), tf.linalg.matvec(tf.linalg.matmul(self.B,  K), y), tf.linalg.matvec(self.B, g))
            # return(ydot)

        # for _ in range(10):
        with tf.GradientTape() as tape:
            tape.watch((Kprop, gainer))

            # error = self.simulate()
            # self.simulate()

            # define the differential system
            output = tfp.math.ode.BDF().solve(func, tf.constant(0), self.x0,
                solution_times = self.times,
                constants = {'K': Kprop, 'g': gainer}
            )

            # caluclate errors
            error = tf.reduce_sum(tf.linalg.normalize(output.states - self.x_targ, axis = 1)[1])
            print(error)
        
        loss_params = tape.gradient(error, [Kprop, gainer])

        Kprop -= learning_rate * loss_params[0]
        gainer -= learning_rate * loss_params[1]


if __name__ == "__main__":

    A = tf.constant(
        [[1,   0,  10],
         [-1,  1,  0],
         [0,   0,  1]],
        dtype=tf.float64)

    B = tf.constant(
        [[1,   -10,    0],
         [0,   1,  0],
         [-1,  0,  1]],
        dtype=tf.float64)

    Kp =  tf.Variable(-1 * tf.constant(
        [[3,  0,  0], 
         [0,   3,  0],
         [0,   0,  3]],
         dtype=tf.float64))

    gain = tf.Variable(tf.constant([-40, 10, -56], dtype = tf.float64))

    target = tf.constant([3, 4, 5], dtype=tf.float64)
    # gain = np.linalg.inv(B) @ (-(A + B @ Kp) @ target)



    # print(np.linalg.eigvals(A + np.matmul(B, Kp)))
    # print(np.linalg.eigvals(A + np.matmul(B, Kp) + B @ gain))
    # print(-np.linalg.inv((A + B @ Kp)) @ (B @ gain))

    system = LinearSystem(A, B, tf.constant([4, 4, 4], dtype=tf.float64), 10, Kp = Kp, x_targ = target, g = gain)
    # system.simulate()

    system.tuning(0.05)

    # plt.plot(system.time, system.output)
    # plt.show()
    # print(system.time)
    # print(out)