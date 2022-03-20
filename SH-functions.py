#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:06:01 2021

@author: yao
"""
from random import randint
import numpy as np
from mpi4py import MPI
from Crypto.Protocol.SecretSharing import Shamir
from Crypto.Hash import SHA256

comm = MPI.COMM_WORLD


def get_n(q, t):
    
    if t == 0:
        n = 0
    else:
        n = [randint(0, q-1) for i in range(0, t)]
    
    return n


# def sharekey(z, w, u, n1, rank, d, t, threshold, users):
#
#     wi = w.reshape(n1, u)
#
#     n = np.arange(1, n1+1).tolist()
#
#     rank_id = n.index(rank)
#
#     share_data = z@wi[rank_id]
#
#     shamir_share = Shamir.split(threshold, users, share_data)
#
#     return shares_recv
