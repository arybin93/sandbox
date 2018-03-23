import time


def how_many_time(original_function):
    def new_function():
        print('Start')
        start_time = time.time()
        result = original_function()
        finish_time = time.time() - start_time
        print('Time: {0:.3f}.'.format(finish_time))
        return result

    return new_function
