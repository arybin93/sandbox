from tasks import get_format_file, get_rectangle_area, merge_files
import time

def main():
    print('Start')
    start_time = time.time()
    # current task

    # Okhotsk Sea
    #get_rectangle_area(lon_start=135.0, lat_start=40.0, lon_end=165.0, lat_end=65.0)
    # Japan Sea
    #get_rectangle_area(lon_start=126.5, lat_start=34.0, lon_end=140.0, lat_end=50.0)
    # Bering Sea, SQ4
    #get_rectangle_area(lon_start=157, lat_start=51.0, lon_end=210.0, lat_end=71.0)
    # Bering Sea, SQ1
    #get_rectangle_area(lon_start=157, lat_start=51.0, lon_end=210.0, lat_end=71.0)

    # open all files, concatenate unique points
    '''
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
    merge_files(list_files_jan)
    '''
    print('Done')
    finish_time = time.time() - start_time
    print('Time: {0:.3f}.'.format(finish_time))

if __name__ == '__main__':
    main()
