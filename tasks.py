import functions
import numpy as np
"""
    Get file for calculation coeffs, for one point
    Output files formats:
        lat    lon    depth_data    z_data    temp_data    salin_data    sigma_data(rho - 1000)    bvf_data
"""
LON = 35.41
LAT = 44.66


def get_format_file():
    # get rho and depth
    data = functions.read_file('depth_rho.txt')

    depth = data[:, 0]
    rho = data[:, 1]

    # reverse arrays
    depth = depth[::-1]
    rho = rho[::-1]

    rho = np.insert(rho, 0, rho[0])
    depth = np.insert(depth, 0, 0.0)

    # calculate bfv
    bvf_for_point = functions.get_bvf(rho, depth)

    # add max depth
    len_data = depth.shape[0]
    max_depth_for_point = np.empty(len_data)
    max_depth_for_point.fill(np.max(depth))

    # fill values
    lon = np.empty(len_data)
    lat = np.empty(len_data)
    sal = np.empty(len_data)
    temp = np.empty(len_data)

    lon.fill(LON)
    lat.fill(LAT)
    sal.fill(0.0)
    temp.fill(0.0)

    # save result to file
    data_for_save = np.c_[lon, lat, max_depth_for_point, depth, temp,  sal, rho, bvf_for_point]
    functions.write_file_pre_coeffs('result.txt', data_for_save)


def get_rectangle_area(lon_start, lat_start, lon_end, lat_end):
    """
    Get rectangle area
    """

    # load file
    data = functions.read_file('GDEM_JUL_SQ1_F_BVF', path=r'D:\ScientificWork\Work\GDEM_BVF_0\JUL', skip_rows=1)
    lat = data[:, 0]
    lon = data[:, 1]
    depth = data[:, 2]
    z_down = data[:, 3]
    temp = data[:, 4]
    sal = data[:, 5]
    rho = data[:, 6]
    n_freq = data[:, 7]

    res_lon = []
    res_lat = []
    res_depth = []
    res_z_down = []
    res_temp = []
    res_sal = []
    res_rho = []
    res_n_freq = []

    # filter result
    for i, x in np.ndenumerate(lat):
        print(i)
        if lat_start <= abs(lat[i]) <= lat_end and lon_start <= abs(lon[i]) <= lon_end:
            res_lat.append(lat[i])
            res_lon.append(lon[i])
            res_depth.append(depth[i])
            res_z_down.append(z_down[i])
            res_temp.append(temp[i])
            res_sal.append(sal[i])
            res_rho.append(rho[i])
            res_n_freq.append(n_freq[i])

    res_lat = np.array(res_lat)
    res_lon = np.array(res_lon)
    res_depth = np.array(res_depth)
    res_z_down = np.array(res_z_down)
    res_temp = np.array(res_temp)
    res_sal = np.array(res_sal)
    res_rho = np.array(res_rho)
    res_n_freq = np.array(res_n_freq)

    # save new file
    data_for_save = np.c_[res_lat, res_lon, res_depth, res_z_down, res_sal, res_temp, res_rho, res_n_freq]
    functions.write_file_pre_coeffs('GDEM_JUL_SQ1_F_BVF_Bering_sea', data_for_save)


def merge_files(files):
    # lat lon  depth  c  alpha  alpha1  beta  z_min_Phi  min_Phi  z_zer_Phi  z_max_Phi  max_Phi
    res_lon = []
    res_lat = []
    res_depth = []
    res_c = []
    res_alpha = []
    res_alpha1 = []
    res_beta = []
    res_z_min_Phi = []
    res_min_Phi = []
    res_z_zer_Phi = []
    res_z_max_Phi = []
    res_max_Phi = []

    for file in files:
        print(file)
        data = functions.read_file(file, path=r'D:\ScientificWork\Work\OkhotskSea\Merge', skip_rows=1)
        lat = data[:, 0]
        lon = data[:, 1]
        depth = data[:, 2]
        c = data[:, 3]
        alpha = data[:, 4]
        alpha1 = data[:, 5]
        beta = data[:, 6]
        z_min_Phi = data[:, 7]
        min_Phi = data[:, 8]
        z_zer_Phi = data[:, 9]
        z_max_Phi = data[:, 10]
        max_Phi = data[:, 11]

        # filter result
        for i, x in np.ndenumerate(lat):
            print(i)

            if lat[i] not in res_lon and lon[i] not in res_lat:
                res_lon.append(lon[i])
                res_lat.append(lat[i])
                res_depth.append(depth[i])
                res_c.append(c[i])
                res_alpha.append(alpha[i])
                res_alpha1.append(alpha1[i])
                res_beta.append(beta[i])
                res_z_min_Phi.append(z_min_Phi[i])
                res_min_Phi.append(min_Phi[i])
                res_z_zer_Phi.append(z_zer_Phi[i])
                res_z_max_Phi.append(z_max_Phi[i])
                res_max_Phi.append(max_Phi[i])

    res_lon_ = np.array(res_lon)
    res_lat_ = np.array(res_lat)
    res_depth_ = np.array(res_depth)
    res_c_ = np.array(res_c)
    res_alpha_ = np.array(res_alpha)
    res_alpha1_ = np.array(res_alpha1)
    res_beta_ = np.array(res_beta)
    res_z_min_Phi_ = np.array(res_z_min_Phi)
    res_min_Phi_ = np.array(res_min_Phi)
    res_z_zer_Phi_ = np.array(res_z_zer_Phi)
    res_z_max_Phi_ = np.array(res_z_max_Phi)
    res_max_Phi_ = np.array(res_max_Phi)

    # save new file
    data_for_save = np.c_[res_lon_, res_lat_, res_depth_, res_c_, res_alpha_, res_alpha1_, res_beta_, res_z_min_Phi_,
                          res_min_Phi_, res_z_zer_Phi_, res_z_max_Phi_, res_max_Phi_]
    functions.write_file_coeffs('GDEM_JAN_KDV_result', data_for_save)


'''

    # Okhotsk Sea
    #get_rectangle_area(lon_start=135.0, lat_start=40.0, lon_end=165.0, lat_end=65.0)
    # Japan Sea
    #get_rectangle_area(lon_start=126.5, lat_start=34.0, lon_end=140.0, lat_end=50.0)
    # Bering Sea, SQ4
    #get_rectangle_area(lon_start=157, lat_start=51.0, lon_end=210.0, lat_end=71.0)
    # Bering Sea, SQ1
    #get_rectangle_area(lon_start=157, lat_start=51.0, lon_end=210.0, lat_end=71.0)

    # open all files, concatenate unique points

    list_files_jul = [
        'GDEM_JUL_KDV_Bering_sea',
        'GDEM_JUL_SQ4_KDV_Japan_sea',
        'GDEM_JUL_SQ4_KDV_Okhotsk_sea',
        'GDEM_JUL_SQ4_KDV_UKM'
    ]

    list_files_jan = [
        'GDEM_JAN_KDV_Bering_sea',
        'GDEM_JAN_SQ4_KDV_Japan_sea',
        'GDEM_JAN_SQ4_KDV_Okhotsk_sea',
        'GDEM_JAN_SQ4_KDV_UKM'
    ]

    #merge_files(list_files_jan)
    import re

    # current task
    with open('data/cards2.txt', "r", encoding="utf8") as file:
        read_data = file.read()
        cards = re.findall(r'\"(.+?)\"', read_data)

        list = []
        for i, card in enumerate(cards):
            if card not in list:
                list.append(card)

    with open('data/cards_result.txt', "w", encoding="utf8") as file_card:
        for card in list:
            file_card.write(card + ',')

 with open('data/file.txt', "r", encoding="utf8") as file:
        read_data = file.read()
        result_accounts = read_data.split('\n')
        result_accounts.pop()

        with open('data/cards.txt', "w", encoding="utf8") as file_card:
            file_card.write('Карта   Владелец   Номер телефона \n\r')
            for index, acc in enumerate(result_accounts):
                values = acc.split('|')
                try:
                    values[1]
                except IndexError:
                    # Not found
                    print('Not found')
                else:
                    account = {}
                    account['phone1'] = values[9]

                    if account['phone1'] != '':
                        pass
                        #print(values[1])
                    else:
                        card_number = int(values[1])
                        if index < 35492:
                            file_card.write(values[1] + ' - ' + values[3] + ' - ' +  values[9] + '\n')

'''