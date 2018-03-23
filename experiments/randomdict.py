import random
import time
from randomdict import RandomDict
from decorators import how_many_time


class RandomDict2(RandomDict):
    def __setitem__(self, key, val):
        if key in self.keys:
            i = self.keys[key]
            self.values[i] = (key, val)
        else:
            self.last_index += 1
            i = self.last_index
            self.values.append((key, val))

        self.keys[key] = i


@how_many_time
def gen_dict():
    r = {}
    for i in range(1000000): r[i] = random.random()
    return r


def exp_random_no_opt():
    r = gen_dict()

    start_time = time.time()
    print(random.choice(list(r.items())))
    finish_time = time.time() - start_time
    print('Time: {0:.5f}.'.format(finish_time))


def exp_random():
    r = RandomDict2()
    start_time = time.time()
    for i in range(1000000): r[i] = random.random()
    finish_time = time.time() - start_time
    print('Time: {0:.5f}.'.format(finish_time))

    start_time = time.time()
    print(r.random_item())
    finish_time = time.time() - start_time
    print('Time: {0:.12f}.'.format(finish_time))


def run():
    exp_random_no_opt()
    exp_random()
