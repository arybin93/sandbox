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
        print('Current mode {}'.format(current_mode))
        c[i] = max_bvf * max_depth / PI / current_mode
        dc = c[i]

        z_zer = np.zeros(max_mode + 1)
        iter = 0
        z = np.linspace(rev_z[0], rev_z[-1], 1000)  # the points of evaluation of solution
        init_cond = [0, 0.01]  # initial value
        while True:
            solver = integrate.ode(sys_phi).set_integrator('dopri5', rtol=1e-4, atol=[1e-6])
            solver.set_f_params(c[i], N)
            solver.set_initial_value(init_cond, rev_z[0])

            # Additional Python step: create vectors to store trajectories
            z_grid = np.zeros((len(z), 1))
            phi = np.zeros((len(z), 1))
            dPhi = np.zeros((len(z), 1))
            phi[0] = 0
            dPhi[0] = 0

            # Integrate the ODE(s) across each delta_z
            zero_counter = 0
            for k in range(1, z.size):
                res = solver.integrate(z[k])

                if not solver.successful():
                    raise RuntimeError("Could not integrate")

                # Store the results
                z_grid[k] = z[k]
                phi[k] = res[0]
                dPhi[k] = res[1]

            #plt.plot(phi, z_grid)
            #plt.gca().invert_yaxis()
            #plt.show()

            n_z_grid = len(z_grid) - 1
            for j in range(1, n_z_grid):
                if phi[j-1] * phi[j] < 0:
                    zero_counter += 1
                    z_zer[zero_counter] = z_grid[j - 1]
            z_zer_Phi = z_zer[i]

            print(iter)
            print('c = {}'.format(c[i]))
            print(np.abs(phi[n_z_grid]/max(phi)))
            print(epsilon_f)
            iter += 1

            if np.abs(phi[n_z_grid]/max(phi)) <= epsilon_f:
                break
            elif dc < 1e-10:
                raise RuntimeError("Could not integrate")
            elif zero_counter <= i:
                dc /= 2
                c[i] -= dc
            elif zero_counter > i:
                dc /= 2
                c[i] += dc
