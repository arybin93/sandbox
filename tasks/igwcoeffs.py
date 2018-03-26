import os
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from constance import PI

num_mode = 1
max_mode = 1

epsilon_f = 1e-7
epsilon_t = 1e-4

epsilon_t_rough = 1e-2
epsilot_t_exact = 1e-3


def read_file(fname, path=r"C:\Users\Rybin\PycharmProjects\sandbox\resources\data", skip_rows=0):
    os.chdir(path)
    print("read file ", fname)
    data = np.loadtxt(fname, skiprows=skip_rows)
    return data


def sys_phi(zz, y, c, N):
    dy = np.zeros(2)
    dy[0] = y[1]
    dy[1] = -((N(zz)/c)**2)*y[0]
    return dy


def calc_coeffs_point():
    data = read_file('data.txt')
    # lat = data[:, 0]
    # lon = data[:, 1]
    # depth = data[:, 2]
    z_down = data[:, 3]
    temp = data[:, 4]
    sal = data[:, 5]
    rho = data[:, 6]
    n_freq = data[:, 7]

    max_depth = np.amax(z_down)
    max_bvf = np.amax(n_freq)
    len_data = np.size(z_down)

    # init new array
    rev_z = np.zeros(len_data)
    rev_bvf = np.zeros(len_data)

    # reverse coordinate z
    for i in range(0, len_data):
        rev_z[len_data-1-i] = max_depth - z_down[i]
        rev_bvf[len_data-1-i] = n_freq[i]

    N = interp1d(rev_z, rev_bvf, kind='cubic')

    c = np.zeros(2)
    for i in range(0, max_mode):
        current_mode = i + 1
        c[i] = max_bvf * max_depth / PI / current_mode

        '''
        while True:
            init_cond = [0, 0.1]
            solver = integrate.ode(sys_phi).set_integrator('dopri5', rtol=1e-4, atol=[1e-6])
            solver.set_f_params(c[i], N)
            solver.set_initial_value(init_cond)



            #plt.plot(phi, z)
            #plt.gca().invert_yaxis()
            #plt.show()

            break
        '''

'''
t0, t1 = 0, 20  # start and end
t = np.linspace(t0, t1, 100)  # the points of evaluation of solution
y0 = [2, 0]  # initial value
y = np.zeros((len(t), len(y0)))  # array for solution
y[0, :] = y0
r = integrate.ode(vdp1).set_integrator("dopri5")  # choice of method
r.set_initial_value(y0, t0)  # initial values
for i in range(1, t.size):
    y[i, :] = r.integrate(t[i])  # get one more value, add it to the array
    if not r.successful():
        raise RuntimeError("Could not integrate")

plt.plot(t, y)
plt.show()
'''
