from system import CartPole, LinSystem
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':


    # linear controllers
    Kp = np.array([[-3,  0,  0], 
                    [0,   -3,  0],
                    [0,   0,  -3]])
    gain = np.array([0, 0, 0])
    x0 = np.array([4, 4, 4, 0, 0, 0])

    # set cost function

    Q = np.array([[2, -1, 0],
                    [-1, 2, -1],
                    [0, -1, 2]])

    R = np.array([[5, -3, 0],   
                    [-3, 5, -2],
                    [0, -2, 5]])


    # '''Test set of single run simulations    '''
    # pend = LinSystem(num = 1, alpha = 0.001, radius = 0.005, t_bound=30)
    # pend.setLinearControl(Kp, gain, x0)
    # pend.setCost(Q, R, gamma = 0.9)

    # output, cost = pend.runSimulation(x0)

    # print(output)
    # plt.plot(pend.times, output[:, 1], label="angle theta")
    # plt.legend()
    # plt.show()

    # '''Quick single agent run to get a representative decay curve '''
    # epochs = 400
    # error = [] 
    # for j in range(epochs):
    #     print("Epoch", j)
    #     err = 0

    #     err = pend.runEpoch(10)
    #     error.append(err)

    #     if j % 100 == 0 and j != 0:
    #         plt.figure(1)
    #         plt.plot(np.linspace(1, j+1, j+1), error, label="single")
    #         plt.title("Testing to the asymptote")
    #         plt.legend(
    # )
    #         plt.show()

    '''Full epoch runs - testing single and multi-agent fully and comparing errors'''
    # setup
    num = 3
    # open a file for records and data plotting later
    f = open("./basic_nonsys_lincont/records.json", "w")
    epochs = 200
    plt.figure(1)

    # single agent runs
    systems = []
    for i in range(num):
        
        sys = LinSystem(num = num)
        sys.setLinearControl(Kp, gain, x0)
        sys.setCost(Q, R)

        systems.append(sys)

    error_sing = []
    for j in range(epochs):
        print("Epoch", j)
        err = 0
        for i in range(num):
            err += systems[i].runEpoch(10)
        
        error_sing.append(err / num)

        
    plt.plot(np.linspace(1, epochs, epochs), error_sing, label="single")

    var = {'error':error_sing}
    for i in range(num):
        var['system' + str(i) + 'Kp'] = systems[i].Kp.tolist()
        var['system' + str(i) + 'gain'] = systems[i].gain.tolist()
    json.dump({'single agent' : var}, f)  

    # multiagent merging by average
    num = 3
    systems = []
    for i in range(num):

        sys = LinSystem(num = num)
        sys.setLinearControl(Kp, gain, x0)
        sys.setCost(Q, R)

        systems.append(sys)

    error_multi = []
    for j in range(epochs):
        print("Epoch", j)
        err = 0
        K_avg = np.zeros(Kp.shape, dtype=float)
        for i in range(num):
            err += systems[i].runEpoch(10)
            K_avg += systems[i].Kp
        
        K_avg = K_avg / num

        for i in range(num):
            systems[i].Kp = K_avg

        error_multi.append(err / num)

    plt.plot(np.linspace(1, epochs, epochs), error_multi, label="multi-avg")

    var = {'error':error_multi}
    for i in range(num):
        var['system' + str(i) + 'Kp'] = systems[i].Kp.tolist()
        var['system' + str(i) + 'gain'] = systems[i].gain.tolist()
    json.dump({'multi agent avg' : var}, f)  

    # multiagent merging by minimum
    num = 3
    systems = []
    for i in range(num):

        sys = LinSystem(num = num)
        sys.setLinearControl(Kp, gain, x0)
        sys.setCost(Q, R)

        systems.append(sys)

    error_multi = []
    for j in range(epochs):
        print("Epoch", j)
        err = []
        K_sets = []
        for i in range(num):
            err.append(systems[i].runEpoch(10))
            K_sets.append(systems[i].Kp)
        
        K_avg = K_sets[np.argmin(err)]
        # K_avg = K_avg / num

        for i in range(num):
            systems[i].Kp = K_avg

        error_multi.append(np.average(err))

    plt.plot(np.linspace(1, epochs, epochs), error_multi, label="multi-min")

    var = {'error':error_multi}
    for i in range(num):
        var['system' + str(i) + 'Kp'] = systems[i].Kp.tolist()
        var['system' + str(i) + 'gain'] = systems[i].gain.tolist()
    json.dump({'multi agent min' : var}, f)  

    # set up the error plot
    plt.xlabel("Epoch #")
    plt.ylabel("Cost")
    plt.title("Testing single vs multiagent learning")
    plt.legend()
    plt.show()

    f.close()
    print('done')

'''
    times = np.arange(0, 10, 0.1)
    output = np.array(pend.setDynamics(x0, times))
    output, cost = pend.runSimulation(x0)

    print(output)
    plt.plot(pend.times, output[:, 1], label="angle theta")
    plt.legend()
    plt.show()'''