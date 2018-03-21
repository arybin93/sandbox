import os
import numpy as np
import matplotlib.pyplot as plt
from constance import GG

# read any file
def read_file(fname, path=r"C:\Users\Rybin\PycharmProjects\science\data", skip_rows=0):
    os.chdir(path)
    print("read file ", fname)
    data = np.loadtxt(fname, skiprows=skip_rows)
    return data


def write_file(fname, data):
    os.chdir(r"D:\\ScientificWork\Work\SpainData\result_data")
    print("write file ", fname)
    np.savetxt(fname, data, '%10.5f    %10.5f    %2.2f    %2.2f    %10.5f    %10.5f    %10.5f    %10.5f')


def write_file_pre_coeffs(fname, data):
    """
    Write file for calculation coeffs
    Output files formats:
            lat    lon    depth_data    z_data    temp_data    salin_data    sigma_data(rho - 1000)    bvf_data
    """
    os.chdir(r"C:\Users\Rybin\PycharmProjects\science\results")
    print("write file ", fname)
    header = 'LAT   LON   DEPTH   Z_DOWN    T    S    RHO    N'
    np.savetxt(fname, data, '%10.5f    %10.5f    %2.2f    %2.2f    %10.5f    %10.5f    %10.5f    %10.5f', header=header)


def write_file_coeffs(fname, data):
    """
    Write file for calculation coeffs
    Output files formats:
            lat    lon    depth_data    z_data    temp_data    salin_data    sigma_data(rho - 1000)    bvf_data
    """
    os.chdir(r"C:\Users\Rybin\PycharmProjects\science\results")
    print("write file ", fname)
    header = 'lon lat depth  c  alpha  alpha1  beta  z_min_Phi  min_Phi  z_zer_Phi  z_max_Phi  max_Phi'
    np.savetxt(fname, data, '%10.5f    %10.5f    %2.2f    %10.5f'
                            '    %10.5f    %10.5f    %10.5f    %10.5f    %10.5f    %10.5f    %10.5f    %10.5f', header=header)


# Foffonoff state sea water
def get_rho(temp, sal):
    print("get rho")
    R00 = 1000
    A = 999.842594 + 6.793952e-2*temp - 9.09529e-3*temp**2 + 1.001685e-4*temp**3 - 1.120083e-6*temp**4 + 6.536332e-9*temp**5
    B = 8.24493e-1 - 4.0899e-3*temp + 7.6438e-5*temp**2 - 8.2467e-7*temp**3 + 5.3875e-9*temp**4
    C = -5.72466e-3  +  1.0227e-4*temp - 1.6546e-6*temp**2
    D = 4.8314e-4
    E = 19652.21 + 148.4206*temp - 2.327105*temp**2 + 1.360477e-2*temp**3 - 5.155288e-5*temp**4
    F = 54.6746-.603459*temp  + 1.09987e-2*temp**2  -  6.167e-5*temp**3
    G = 7.944e-2+1.6483e-2*temp - 5.3009e-4*temp**2
    H = 3.239908  +  1.43713e-3*temp  +  1.16092e-4*temp**2 - 5.77905e-7*temp**3
    I = 2.2838e-3 - 1.0981e-5*temp - 1.6078e-6*temp**2
    J = 1.91075e-4
    M = 8.50935e-5 - 6.12293e-6*temp + 5.2787e-8*temp**2
    N = -9.9348e-7 + 2.0816e-8*temp +  9.1697e-10*temp**2
    R0 = A + B*sal + C*sal**1.5 + D*sal**2
    P = 0
    RK = E+F*sal+G*sal**1.5+(H+I*sal+J*sal**1.5)*P+(M+N*sal)*P**2
    rho = R0/(1-P/RK)-R00
    return rho


# get Brenta-Vaisala frequency
def get_bvf(rho, depth):
    length = rho.shape[0]

    # init bvf array
    bvf = np.empty(length)
    bvf.fill(0)

    # incompressible fluid case
    r1 = rho[0]
    for i in range(length-1):
        r2 = rho[i+1]
        bvf[i] = np.sqrt(
            np.abs(
                GG*(r2 - r1)/(depth[i+1] - depth[i])/(1000 + r2)
            )
        )
        r1 = r2

    bvf[i+1] = bvf[i]

    return bvf



'''

# -*- coding: utf-8 -*-
"""
Scripts for processing data from Spain shelf

Created on Tue Aug  9 22:21:58 2016

@author: Artem Rybin
@email: arybin93@gmail.com

inputs files formats:
During_storm_P1_salinity.dat
    LON    DEPTH    salinity
-9.55848    -32.52     35.74

output files formats:
lat    lon    depth_data    z_data    temp_data    salin_data    sigma_data(rho - 1000)    bvf_data
"""


'''

def get_index_point(array):
    un_array = np.unique(array)
    length = un_array.shape[0]

    index_point = np.empty([length, 2]);
    index_point.fill(0)

    for i in range(length):
        ind = np.where(array == un_array[i])[0]
        index_point[i, 0] = ind[0]
        index_point[i, 1] = ind[-1]+1

    return index_point


def spain_task():
    filenamesTemperature = ["Pre_storm_P1_temperature.dat", "During_storm_P1_temperature.dat", "Post_storm_P1_temperature.dat",
                            "Pre_storm_S2_temperature.dat", "During_storm_S2_temperature.dat", "Post_storm_S2_temperature.dat"];

    filenamesSalinity = ["Pre_storm_P1_salinity.dat", "During_storm_P1_salinity.dat", "Post_storm_P1_salinity.dat",
                        "Pre_storm_S2_salinity.dat", "During_storm_S2_salinity.dat", "Post_storm_S2_salinity.dat"];

    for i in range(6):
        dataTemp = read_file(filenamesTemperature[i])
        dataSal = read_file(filenamesSalinity[i])
        depth = abs(dataSal[:,1])
        lon = dataSal[:, 0]
        lat = lon
        temprature = dataTemp[:,2]
        salinity = dataSal[:,2]
        ind_point = get_index_point(lon)
        ind_point = ind_point.astype(int)

        # calculation sigma (rho - 1000)
        rho = get_rho(temprature, salinity)
        length = ind_point.shape[0]

        # init bvf array
        res_bvf = np.array([])
        res_rho = np.array([])
        res_depth = np.array([])
        res_salinity = np.array([])
        res_temperature = np.array([])
        res_lon = np.array([])
        res_lat = np.array([])
        res_max_depth = np.array([])

        for j in range(length):
            j_start = ind_point[j, 0]
            j_end = ind_point[j, 1]
            # get data for every point
            rho_for_point = rho[j_start:j_end]
            depth_for_point = depth[j_start:j_end]
            salinity_for_point = salinity[j_start:j_end]
            temprature_for_point = temprature[j_start:j_end]
            lon_fo_point = lon[j_start:j_end]
            lat_for_point = lat[j_start:j_end]

            # add data for surface, every point
            rho_for_point = np.insert(rho_for_point, 0, rho_for_point[0])
            depth_for_point = np.insert(depth_for_point, 0, 0)
            salinity_for_point = np.insert(salinity_for_point, 0, salinity_for_point[0])
            temprature_for_point = np.insert(temprature_for_point, 0, temprature_for_point[0])
            lon_fo_point = np.insert(lon_fo_point, 0, lon_fo_point[0])
            lat_for_point = np.insert(lat_for_point, 0, lat_for_point[0])

            # add max depth
            len_point_data = depth_for_point.shape[0]
            max_depth_for_point = np.empty(len_point_data);
            max_depth_for_point.fill(np.max(depth_for_point))

            res_rho = np.hstack((res_rho, rho_for_point))
            res_depth = np.hstack((res_depth, depth_for_point))
            res_salinity = np.hstack((res_salinity, salinity_for_point))
            res_temperature = np.hstack((res_temperature, temprature_for_point))
            res_lon = np.hstack((res_lon, lon_fo_point))
            res_lat = np.hstack((res_lat, lat_for_point))
            res_max_depth = np.hstack((res_max_depth, max_depth_for_point))

            # get bvf for every point
            bvf_for_point = get_bvf(rho_for_point, depth_for_point)
            res_bvf = np.hstack((res_bvf, bvf_for_point))

        # save result to file
        data_for_save = np.c_[res_lon, res_lat , res_max_depth, res_depth, res_temperature, res_salinity, res_rho, res_bvf]
        fname = filenamesTemperature[i].replace("temperature", "data")
        write_file(fname, data_for_save)

    print("Done")
