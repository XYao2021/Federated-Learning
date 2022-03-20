#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:06:01 2021

@author: yao
"""
from random import randint
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD


def get_n(q, d, U, T):
    
    if T == 0:
        n = 0
    else:
        n = [[randint(0,q-1) for i in range(0,int(d/(U-T)))] for i in range(0,T)]
    
    return n

def sharekey(Z,W,U,N1,rank,d,T):

    Wi = W.reshape(N1,U)
    
    N = np.arange(1,N1+1).tolist()
    
    rank_id = N.index(rank)
    
    Shares = [[0 for i in range(0,int(d/(U-T)))] for i in range(0,N1)]
    x = [[0 for i in range(0,U)] for i in range(0,int(d/(U-T)))]

    for i in range(0,int(d/(U-T))):
        
        for j in range(0,N1):
            he = 0
            for k in range(0,U):
                
                x[i][k] = Z[k][i]
                he = he + x[i][k]*Wi[j][k]
                   
            Shares[j][i] = he
    
    Send_set = N.copy()
    Send_set.remove(rank)
    
    Send_Shares = Shares.copy()
    Send_Shares.remove(Send_Shares[rank_id])
    
    Shares_recv = [0 for i in range(0,N1-1)]
    
    for i in range(0,len(Send_set)):
        
        comm.send(Send_Shares[i],dest=Send_set[i],tag=2)
        Shares_recv[i] = comm.recv(source=Send_set[i],tag=2)
    
    Shares_recv.insert(rank_id,Shares[rank_id])
        
    return Shares_recv