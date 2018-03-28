import os
import numpy as np
from scipy import integrate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from constance import PI
from scipy.integrate import odeint

num_mode = 1
max_mode = 1

epsilon_f = 1e-4
epsilon_t = 1e-4

epsilon_t_rough = 1e-2
epsilot_t_exact = 1e-3


def read_file(fname, path=r"C:\Users\Rybin\PycharmProjects\sandbox\resources\data", skip_rows=0):
    os.chdir(path)
    print("read file ", fname)
    data = np.loadtxt(fname, skiprows=skip_rows)
    return data


def sys_phi(y, z, N, c):
    phi, dphi = y
    dy = [dphi, -((N(z) / c) ** 2) * phi]
    return dy


def func_r(z, c, alpha, dif_phi, dif2_phi):
    rhs = (-alpha/c + 3 * dif_phi(z)* dif2_phi(z))
    return rhs


def sys_f_or_tn(y, z, N, c, alpha, dif_phi, dif2_phi):
    f, df = y
    dy = [df, -((N(z) / c) ** 2) * f + func_r(z, c, alpha, dif_phi, dif2_phi)]
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

    N = interp1d(rev_z, rev_bvf, kind='cubic', fill_value="extrapolate")

    c = np.zeros(2)
    beta = np.zeros(2)
    alpha = np.zeros(2)
    for i in range(0, max_mode):
        current_mode = i + 1
        print('Current mode {}'.format(current_mode))
        c[i] = max_bvf * max_depth / PI / current_mode
        dc = c[i]

        z_zer = np.zeros(max_mode + 1)
        z = np.linspace(rev_z[0], rev_z[-1], 100000)  # the points of evaluation of solution
        init_cond = [0, 0.01]  # initial value

        while True:
            zero_counter = 0
            sol = odeint(sys_phi, init_cond, z, args=(N, c[i]), rtol=[1e-4, 1e-4], atol=[1e-6, 1e-6])
            phi = sol[:, 0]
            dphi = sol[:, 1]

            n_z_grid = len(z) - 1
            for j in range(1, n_z_grid):
                if phi[j - 1] * phi[j] < 0:
                    zero_counter += 1
                    z_zer[zero_counter] = z[j - 1]
            z_zer_Phi = z_zer[i]

            if np.abs(phi[-1] / np.max(phi)) <= epsilon_f:
                break
            elif dc < 1e-10:
                raise RuntimeError("Could not integrate")
            elif zero_counter <= i:
                dc /= 2
                c[i] -= dc
            elif zero_counter > i:
                dc /= 2
                c[i] += dc

        min_phi, ind_min_phi = np.min(phi), np.argmin(phi)
        # second mode, pronounced minimum
        if np.abs(min_phi) > 10**(-1):
            phi = phi * (-1)
            dphi = dphi * (-1)

        max_phi, ind_max_phi = np.max(phi), np.argmax(phi)
        min_phi, ind_min_phi = np.min(phi), np.argmin(phi)

        # normalization
        phi /= max_phi
        dphi /= max_phi

        min_phi /= max_phi
        max_phi /= max_phi

        z_max_Phi = z[ind_max_phi]
        z_min_Phi = z[ind_min_phi]

        # coeffs
        num_beta = phi*phi
        denom = dphi*dphi
        num_alpha = denom*dphi

        beta[i] = (c[i]/2) * np.trapz(num_beta, z)/np.trapz(denom, z)
        alpha[i] = (3*c[i]/2) * np.trapz(num_alpha, z)/np.trapz(denom, z)

        print(c)
        print(beta)
        print(alpha)

        # alpha 1
        t_end_prev = 0.0
        init_cond = [0, 0.01]  # initial value

        dif_phi = interp1d(z, dphi, kind='cubic', fill_value="extrapolate")
        d2phi = numpy.gradient(dphi, z)
        dif2_phi = interp1d(z,d2phi, kind='cubic', fill_value="extrapolate")

        while True:
            sol = odeint(sys_f_or_tn, init_cond, z, args=(N, c[i], alpha[i], dif_phi, dif2_phi),
                         rtol=[1e-4, 1e-4], atol=[1e-6, 1e-6])
            t = sol[:, 0] # F or Tn
            dt = sol[:, 1] # dF or dTn
            # TODO

        plt.plot(phi, z)
        plt.gca().invert_yaxis()
        plt.show()


def sys_phi_new(z, y, c, N):
    phi, dphi = y
    dy = [dphi, -((N(z) / c) ** 2) * phi]
    return dy


def calc_coeffs_point_new():
    data = read_file('data.txt')
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

    N = interp1d(rev_z, rev_bvf, kind='cubic', fill_value="extrapolate")

    c = np.zeros(2)
    for i in range(0, max_mode):
        current_mode = i + 1
        print('Current mode {}'.format(current_mode))
        c[i] = max_bvf * max_depth / PI / current_mode
        dc = c[i]

        z_zer = np.zeros(max_mode + 1)
        iter = 0
        z = np.linspace(rev_z[0], rev_z[-1], 1500)  # the points of evaluation of solution
        init_cond = [0, 0.01]  # initial value

        while True:
            solver = integrate.ode(sys_phi_new).set_integrator('dopri5', rtol=1e-4, atol=[1e-6])
            solver.set_f_params(c[i], N)
            solver.set_initial_value(init_cond, rev_z[0])

            # Additional Python step: create vectors to store trajectories
            phi = np.zeros((len(z), 1))
            dPhi = np.zeros((len(z), 1))

            # Integrate the ODE(s) across each delta_z
            zero_counter = 0
            for k in range(1, z.size):
                res = solver.integrate(z[k])

                if not solver.successful():
                    raise RuntimeError("Could not integrate")

                # Store the results
                phi[k] = res[0]
                dPhi[k] = res[1]

            n_z_grid = len(z) - 1
            for j in range(1, n_z_grid):
                if phi[j - 1] * phi[j] < 0:
                    zero_counter += 1
                    z_zer[zero_counter] = z[j - 1]
            z_zer_Phi = z_zer[i]

            print(iter)
            print('c = {}'.format(c[i]))
            print(np.abs(phi[-1] / np.max(phi)))
            iter += 1

            if np.abs(phi[-1] / np.max(phi)) <= epsilon_f:
                break
            elif dc < 1e-10:
                raise RuntimeError("Could not integrate")
            elif zero_counter <= i:
                dc /= 2
                c[i] -= dc
            elif zero_counter > i:
                dc /= 2
                c[i] += dc

    plt.plot(phi, z)
    plt.gca().invert_yaxis()
    plt.show()
