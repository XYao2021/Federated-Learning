#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:06:19 2021

@author: yao
"""
from mpi4py import MPI
from functions import get_n,sharekey
from random import randint,choice
import numpy as np
import time
import sys

Start_time = time.time()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"----------------------------------- Offline --------------------------------------"

q = 7
N = 20
T = 0
Drop_rate = 0.3
U = int(N-Drop_rate*N)
d = 2*(U-T)

np.seterr(divide='ignore', invalid='ignore')
column = np.arange(U+1,U+N+1).reshape(-1,1)
row = np.arange(1,U+1)
W = (1.0)/(column-row)

if rank != 0:
    
    z = [randint(1,q-1) for i in range(0,int(d))]

    zi = [z[2*i:(2*i+(int(d/(U-T))))] for i in range(0,U)]
    n = get_n(q,d,U,T)
    
    Z = [0 for i in range(0,U)]
    
    for i in range(0,U-T):
        
        Z[i] = zi[i]
        
    for i in range(U-T,U):
        
        Z[i] = n[i-U+T]
    
    Shares_recv = sharekey(Z,W,U,N,rank,d,T)
 
Offline_End_time = time.time()
"---------------------------------- Round 1 ----------------------------------------"

y = []
if rank != 0:
    
    xu = np.arange(100000, dtype=np.int64)

    y = [0 for i in range(0,len(xu))]

    for i in range(0,len(xu)):
        
        y[i] = xu[i] + sum(z)

yu = comm.gather(y,root=0)
comm.Barrier()

if rank == 0:

    yu.remove(yu[0])
    
    U0 = list(np.arange(1,N+1))
    
    Drop = int(N*Drop_rate)
    Drop_user = [0 for i in range(0,Drop)]
    Drop_id = [0 for i in range(0,Drop)]
    U1 = U0.copy()
    yu_remain = yu.copy()

    for i in range (0,Drop):
        
        Drop_user[i] = choice(U1)
        Drop_id[i] = U0.index(Drop_user[i])
        U1.remove(Drop_user[i])
        yu_remain.remove(yu[Drop_id[i]])
        
    yu_sum = [0 for i in range(0,len(yu_remain[0]))]
    
    for i in range(0,len(yu_remain[0])):
        
        he = 0
        
        for j in range(0,U):
            
            he = he + yu_remain[j][i]
            
        yu_sum[i] = he
    
else:
    
    U1=[]
 
U1 = comm.bcast(U1,root=0)
comm.Barrier() 

R1_End_time = time.time()
"---------------------------------- Round 2 ----------------------------------------"

Sum = 0
if rank != 0:

    if rank in U1:

        U0 = list(np.arange(1,N+1))
        Drop_user = [item for item in U0 if item not in U1]
        
        Shares_remain = Shares_recv.copy()
        Drop_id = [0 for i in range(0,len(Drop_user))]
        
        for i in range(0,len(Drop_user)):
            
            Drop_id[i] = U0.index(Drop_user[i])
            Shares_remain.remove(Shares_recv[Drop_id[i]])

        Sum = [0 for i in range(0,int(d/(U-T)))]
        
        for i in range(0,int(d/(U-T))):
            
            he = 0
            for j in range(0,U):
                
                he = he + Shares_remain[j][i]
            Sum[i] = he


    else:
        
        None

Sum = comm.gather(Sum,root=0)
comm.Barrier()

if rank == 0:

    Sum = [i for i in Sum if i != 0]

    Wi = W.reshape(N,U).tolist()
    
    Wi_remain = Wi.copy()

    for i in range(0,len(Drop_user)):
        
        Drop_id = U0.index(Drop_user[i])

        Wi_remain.remove(Wi[Drop_id])
    
    x = [[0 for i in range(0,U)] for i in range(0,int(d/(U-T)))]
    Sum_adjust = [[0 for i in range(0,U)] for i in range(0,int(d/(U-T)))]
    
    for i in range(0,U):
        
        for j in range(0,int(d/(U-T))):
            
            Sum_adjust[j][i] = Sum[i][j]
    
    for i in range(0,int(d/(U-T))):
        
        x[i] = np.linalg.solve(Wi_remain,Sum_adjust[i]) 
    
    he = [0 for i in range(0,int(d/(U-T)))]
    
    for i in range(0,int(d/(U-T))):
        
        h = 0
        
        for j in range(0,U):
            
            h = h + x[i][j]
            
        he[i] = round(h)

    xu_sum = [0 for i in range(0,len(yu_sum))]

    for i in range(0,len(yu_sum)):
        
        xu_sum[i] = yu_sum[i] - sum(he)
    
    # print('this is Final Sum :',xu_sum)
    # sys.stdout.flush()
    
comm.Barrier()    

End_time = time.time()
print('User',rank,':','\n',
      'Offline time cost:',Offline_End_time-Start_time,'s','\n',
      'R1 time cost:',R1_End_time-Offline_End_time,'s','\n',
      'R2 time cost:',End_time-R1_End_time,'s','\n',
      'total time cost:',End_time-Start_time,'s','\n',
      'THE END','\n')
sys.stdout.flush()