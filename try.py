#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:06:19 2021

@author: yao
"""
import mpi4py as MPI
from functions import get_n
from random import randint, choice
import numpy as np
import time
import itertools
import sys
from math import comb

Start_time = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"----------------------------------- Offline --------------------------------------"

q = 7
N = 28
T = 0
# Drop_rate = 0.3
# U = int(N-Drop_rate*N)
U = 10
Org_set = list(np.arange(1, N+1))

"---generate cauchy matrix---"

matrix_dict = {}
i_iter = 0

while i_iter < int(N - U + 1):
    np.seterr(divide='ignore', invalid='ignore')
    column = np.arange(U + 1, U + 1 + N - i_iter).reshape(-1, 1)
    row = np.arange(1, U + 1)
    W = 1.0 / (column - row)
    matrix_dict[N - i_iter] = W
    i_iter += 1

# print(matrix_dict, '\n',  matrix_dict.get(10))

if rank != 0:

    z = [randint(1, q-1) for i in range(0, U-T)]

    Send_set = Org_set.copy()
    Send_set.remove(rank)

    "---recv original data---"

    recv_data = []
    for i in range(0, len(Send_set)):

        comm.send(z, dest=Send_set[i], tag=1)
        recv_data.append(comm.recv(source=Send_set[i], tag=1))

    recv_data.insert(Org_set.index(rank), z)
    # print(len(recv_data))

    "---prepare all Zi Data---"

    my_list = []

    for L in range(0, (len(Org_set)+1)):
        count = 0
        for subsets in itertools.combinations(Org_set, L):
            element = list(subsets)
            if len(element) >= U:
                if rank in element:
                    my_list.append(element)
                else:
                    None
            else:
                None

    # print(rank, 'this is my_list: ', my_list, len(my_list), '\n')
    # sys.stdout.flush()

    Z = []

    for i in range(0, len(my_list)):

        work_set = my_list[i]
        work_data = []
        for j in range(0, len(work_set)):
            idx = Org_set.index(work_set[j])
            work_data.append(recv_data[idx])
        # print(i, 'this is work set: ', work_set, len(work_set), '\n')
        # print('this is org: ', recv_data, len(recv_data), '\n')
        # print(i, 'this is work data: ', work_data, len(work_data), '\n')
        Sum_vector = sum(np.array(work_data))
        W_work = matrix_dict.get(len(work_set))
        Zi = W_work@Sum_vector
        # print(i, 'this is sum V: ', Sum_vector, len(Sum_vector), len(work_data[0]), '\n')
        # print(i, 'this is use W: ', W_work, len(W_work), '\n')
        # print(i, 'this is Zi: ', Zi, len(Zi), '\n')
        sys.stdout.flush()
        Z.append(Zi)
    # print(rank, 'this is Z: ', Z, len(Z), '\n')
    sys.stdout.flush()


Offline_End_time = time.time()

"---------------------------------- Round 1 ----------------------------------------"

y = []
if rank != 0:

    xu = np.arange(1, 100001, dtype=np.int64)

    y = [0 for i in range(0, len(xu))]

    for i in range(0, len(xu)):

        y[i] = xu[i] + sum(z)

yu = comm.gather(y, root=0)
comm.Barrier()

if rank == 0:

    yu.remove(yu[0])

    # Drop = int(N*Drop_rate)
    Drop = N - U

    Drop_user = [0 for i in range(0, Drop)]
    Drop_id = [0 for i in range(0, Drop)]

    U1 = Org_set.copy()
    yu_remain = yu.copy()

    for i in range(0, Drop):

        Drop_user[i] = choice(U1)
        Drop_id[i] = Org_set.index(Drop_user[i])
        U1.remove(Drop_user[i])
        yu_remain.remove(yu[Drop_id[i]])

    yu_sum = [0 for i in range(0, len(yu_remain[0]))]

    for i in range(0, len(yu_remain[0])):

        he = 0

        for j in range(0, len(U1)):

            he = he + yu_remain[j][i]

        yu_sum[i] = he

else:

    U1 = []

U1 = comm.bcast(U1, root=0)
comm.Barrier()

R1_End_time = time.time()
"---------------------------------- Round 2 ----------------------------------------"
Z_useful = 0

if rank != 0:

    if rank in U1:

        # print(rank, 'this is U1: ', U1, len(U1), '\n')
        # sys.stdout.flush()
        U1_id = my_list.index(U1)
        # print(rank, 'this is index of U1: ', U1_id, '\n')
        # sys.stdout.flush()
        Useful_data = Z[U1_id]
        Z_useful = Useful_data[U1.index(rank)]
        # print(rank, 'this is Useful_data: ', Useful_data, len(Useful_data), '\n')
        # sys.stdout.flush()
        # print(rank, 'this is Z_useful: ', Z_useful, '\n')

Z_useful = comm.gather(Z_useful, root=0)
comm.Barrier()

if rank == 0:

    # print(rank, 'this is Z_useful', Z_useful, '\n')
    Z_useful = [i for i in Z_useful if i != 0]
    # print(rank, 'this is new Z_useful', Z_useful, '\n')

    W_final = matrix_dict.get(len(U1))
    # print(W_final)

    X = np.linalg.solve(W_final, Z_useful)
    # print('this is X: ', X)

    Z_sum = 0
    for i in range(0, len(X)):
        Z_sum = Z_sum + round(X[i])
    # print(Z_sum)

    for i in range(0, len(yu_sum)):

        yu_sum[i] = yu_sum[i] - Z_sum

    print(yu_sum[0:10])

comm.Barrier()

End_time = time.time()
print('User', rank, ':', '\n',
      'Offline time cost:', Offline_End_time-Start_time, 's', '\n',
      'R1 time cost:', R1_End_time-Offline_End_time, 's', '\n',
      'R2 time cost:', End_time-R1_End_time, 's', '\n',
      'total time cost:', End_time-Start_time, 's', '\n',
      '\n')
sys.stdout.flush()
