import multiprocessing as mp
import time
import os
import contextlib
import sys
from functools import partial
class foo:
    bar = 1

class foo1:
    bar = 1
f_obj = foo()

f1_obj = foo1()

def f(f_obj, x):
    global f1_obj
    f1_obj = 
    prod = 1
    for i in range(int(10e4)):
        prod *= 2
    # print('f bar is ', f_obj.bar)
    print('f1 bar is ', f1_obj.bar)
    print(os.getpid())
    f_obj.bar += 1
    f1_obj.bar += 1
    return x*x


def a():
    with Pool(processes=4) as pool:

        # print "[0, 1, 4,..., 81]"
        print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20,))      # runs in *only* one process
        print(res.get(timeout=1))             # prints "400"

        # evaluate "os.getpid()" asynchronously
        res = pool.apply_async(os.getpid, ()) # runs in *only* one process
        print(res.get(timeout=1))             # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])

        # make a single worker sleep for 10 secs
        res = pool.apply_async(time.sleep, (10,))
        try:
            print(res.get(timeout=1))
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more work")


if __name__ == '__main__':
    # start 4 worker processes

    mp.set_start_method('spawn')
    func = partial(f, f_obj)
    with mp.Pool(processes=4) as pool:
        print(pool.map(partial(f, f_obj), range(10)))
    # exiting the 'with'-block has stopped the pool