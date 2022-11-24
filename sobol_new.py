#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:09:42 2019

@author: jmadriga
"""

# -*- coding: utf-8 -*-
"""sobol_point_gen.py

 Usage:
    sobol_point_gen.py N D <dimensions_file>
    sobol_point_gen.py -h | --help

 Options:
     -h --help     Show this screen.     

 Arguments:
     N  Number of points to generate (<2**32)
     D  Number of dimensions for each point. (D <= max D in dimensions_file)
     dimensions_file    See: https://web.maths.unsw.edu.au/~fkuo/sobol/

 Created from an original c++ source by Frances Y. Kuo and Stephen Joe

Created on Fri Nov  1 09:13:12 2019

@author: Paddy3118
"""

from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt



#%%
def binhi1(b):
    "highest 1 bit of binary b. (== ceil(log2(b)))"
    pos = 0
    while b > 0:
        pos, b = pos + 1, b // 2
    return pos

def binlo0(b):
    "lowest 0 bit from right of binary b. Always >=1"
    pos, b2 = 1, b // 2
    while b2 * 2 != b:
        pos, b, b2 = pos + 1, b2, b2 // 2
    return pos

#%%
def direction_file_reader(d, fname="Sobol_new-joe-kuo-6.21201"):
    """\
    Reads d+header lines from direction file of format:
        d       s       a       m_i     
        2       1       0       1 
        3       2       1       1 3 
        4       3       1       1 3 1 
        5       3       2       1 1 1 
        6       4       1       1 1 3 3
        7       4       4       1 3 5 13 
        8       5       2       1 1 5 5 17 
        9       5       4       1 1 5 5 5 
        10      5       7       1 1 7 11 19 
        11      5       11      1 1 5 1 1 
        12      5       13      1 1 1 3 11 
        13      5       14      1 3 5 5 31 
        14      6       1       1 3 3 9 7 49 
        15      6       13      1 1 1 15 21 21 
        16      6       16      1 3 1 13 27 49 
        17      6       19      1 1 1 15 7 5 
        18      6       22      1 3 1 15 13 25 
        19      6       25      1 1 5 5 19 61 
        20      7       1       1 3 7 11 23 15 103 
        21      7       4       1 3 7 13 13 15 69 
        22      7       7       1 1 3 13 7 35 63 
        23      7       8       1 3 5 9 1 25 53 
        24      7       14      1 3 1 13 9 35 107 
        25      7       19      1 3 1 5 27 61 31 
        26      7       21      1 1 5 11 19 41 61 
        27      7       28      1 3 5 3 3 13 69 
        28      7       31      1 1 7 13 1 19 1 
        29      7       32      1 3 7 5 13 19 59 
        30      7       37      1 1 3 9 25 29 41 
        31      7       41      1 3 5 13 23 1 55 
        32      7       42      1 3 7 3 13 59 17 
        ...
    """
    with open(fname) as f:
        headers = f.readline().strip().split()
        assert headers == ['d', 's', 'a', 'm_i']
        directions = []
        for i, line in zip(range(d), f):
            d, s, a, *m = [int(cell) for cell in line.strip().split()]
            directions.append((d, s, a, tuple([0] + m)))
        return directions



def generate_points(N, D, fname="Sobol_new-joe-kuo-6.21201"):
    return np.asarray(list(sobol_point_gen(N, D)))
#%%
@lru_cache(128)
def calc_dnum(s, a, m, mxbits):                      # V, direction numbers
    if mxbits <= s:
        dirnum = [0] + [m[i] << (32 - i) for i in range(1, mxbits+1)]
    else:
        dirnum = [0] + [m[i] << (32 - i) for i in range(1, s+1)]
        for i in range(s+1, mxbits+1):
            dirnum.append(dirnum[i-s] ^ (dirnum[i-s] >> s))
            for k in range(1, s):
                dirnum[i] ^= (((a >> (s-1-k)) & 1) * dirnum[i-k])
    return dirnum

def sobol_point_gen(N, D, fname="Sobol_new-joe-kuo-6.21201"):
    pow2_32 = 2**32
    mxbits = binhi1(N)                              # L, max bits needed
    dnum = [1 << (32- i) for i in range(mxbits+1)]  # V, direction numbers
    directions = direction_file_reader(D, fname)
    xd = [0 for j in range(D)]                      # X
    pt = [0 for j in range(D)]                      # POINTS
    yield pt
    for n in range(1, N):
        nfirst0 = binlo0(n-1)                       # C
        pt = [0 for j in range(D)]
        xd[0] = xd[0] ^ dnum[nfirst0]
        pt[0] = xd[0] / pow2_32
        for jj in range(1, D):
            _, s, a, m = directions[jj-1]
            dnum2 = calc_dnum(s, a, m, mxbits)      # V, direction numbers
            xd[jj] = xd[jj] ^ dnum2[nfirst0]
            pt[jj] = xd[jj]/pow2_32
        yield pt

#%%
if __name__ == '__main__':
    first_sobol_3D_points = [
         [0, 0, 0],
         [0.5, 0.5, 0.5],
         [0.75, 0.25, 0.25],
         [0.25, 0.75, 0.75],
         [0.375, 0.375, 0.625],
         [0.875, 0.875, 0.125],
         [0.625, 0.125, 0.875],
         [0.125, 0.625, 0.375],
         [0.1875, 0.3125, 0.9375],
         [0.6875, 0.8125, 0.4375]]
    generated = list(sobol_point_gen(10, 3))
    assert generated == first_sobol_3D_points
    
