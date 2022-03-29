from mpi4py import MPI
import sys
import struct
import numpy as np
import itertools
import random
from sympy import Matrix

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# global_groups = comm.Get_group()


"constrains: U <= K - U + 1"
K = 3
U = 2
S = 2
L = 10
q = 7
#
org_group = list(np.arange(1, K+1))
all_group = itertools.combinations(org_group, S)
whole_group = [list(i) for i in all_group]

a_null_group = []
if rank != 0:
    send_group = [i for i in whole_group if rank in i]
    print(rank, send_group)
    # null = [i for i in whole_group if i not in send_group]
    # print(rank, 'null: ', null, '\n')
    Zvk = []
    ak = []
    for group in send_group:
        null_group = [i for i in org_group if i not in group]
        print(null_group, group[0])
        if rank in group:
            if rank == group[0]:
                Zv = np.random.randint(1, q, L, dtype=int)
                Zvk.append(Zv)
                b_set = group.copy()
                b_set.remove(rank)
                # print(rank, 'this is org Z: ', Zv, '\n')
                a = np.random.randint(1, q, U, dtype=int)
                a_null = list(Matrix([a]).nullspace()[0])
                print(rank, 'this is a_null: ', a, a_null, '\n')
                ak.append(a)
                for target in b_set:
                    comm.send([a, Zv], dest=target, tag=1)
                for item in null_group:
                    comm.send([a_null], dest=item, tag=2)
            elif rank != group[0]:
                data = comm.recv(source=group[0], tag=1)
                Zvk.append(data[1])
                ak.append(data[0])
        if rank in null_group:
            null_data = comm.recv(source=group[0], tag=2)
            a_null_group.append(null_data)
    # print(rank, 'this is ak: ', ak)
    # sys.stdout.flush()

Xi = []
if rank != 0:
    for i in range(len(Zvk)):
        Zvk[i] = np.split(Zvk[i], U)
    print(rank, 'this is Zvk after adjust: ', Zvk)
    print(rank, 'this is ak: ', ak)
    sys.stdout.flush()
    Wi = np.split(np.random.randint(1, q, L, dtype=int), U)
    print(rank, 'this is Wi: ', Wi)
    sys.stdout.flush()
    Wij = []
    for j in range(len(Wi)):
        he = np.zeros((len(Wi[0]),), dtype=int)
        for k in range(len(Zvk)):
            he += Zvk[k][j]*ak[j][k]
        Wij.append(he)
        Xi.append(Wi[j] + he)
    print(rank, 'this is Xi: ', Xi, '\n')
    sys.stdout.flush()

XU = comm.gather(Xi, root=0)
comm.barrier()

if rank == 0:
    XU.pop(0)
    print(rank, 'this is XU', XU, '\n')
    sys.stdout.flush()
    X_sum = []
    for i in range(len(XU[0])):
        SUM = np.zeros((int(L/U), ), dtype=int)
        for j in range(len(XU)):
            SUM += XU[j][i]
        X_sum.append(SUM)
    print(rank, 'this is X_sum: ', X_sum, '\n')
    sys.stdout.flush()
    U1 = org_group.copy()
    U1.remove(random.sample(org_group, K-U))
    print(rank, 'this is U1: ', U1, '\n')
    sys.stdout.flush()
comm.barrier()

if rank != 0:
    print(rank, 'Wij: ', Wij, '\n')
    sys.stdout.flush()
    print(rank, 'a_null_group: ', a_null_group, '\n')
    sys.stdout.flush()
#     print(rank, Z, '\n')
    # sys.stdout.flush()
    # if rank in group:
    #     Z_recv = new_comm.bcast(Z, root=group[0])

# if rank != 0:
#     print(rank, Z_recv, '\n')
# global_groups = comm.Get_group()
# size = global_groups.Get_size()
# print(size)
#
# g = [1, 2]
# new_group = global_groups.Incl(g)
# new_comm = comm.Create(new_group)
# # print(g[-1])
# a = []
# if rank == 1:
#     data = 12
#     a = new_comm.bcast(data, root=g[0])
# #
# if rank in g:
#     print(rank, a, '\n')
# a = []
# if rank == g[0]:
#     data = [1, 2, 3, 4, 5]
#     print(rank, data)
#     a = new_comm.bcast(data, root=1)
# if rank != 0:
#     print(rank, a, '\n')
#
# if rank != 0:
#     print(rank, a)

# group = [1, 2, 3]
# S = 2
# send_group = itertools.combinations(group, S)
# print(list(send_group))
#
# a = np.random.randint(1, q, L, dtype=int)
# b = np.split(a, U)
# c = [list(i) for i in b]
# print(a)
# print(c)


# MPI.Comm.Bcast()
