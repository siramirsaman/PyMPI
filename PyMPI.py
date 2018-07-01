#!/usr/bin/env python
"""
Python MPI A.x = b
"""

from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

N = 10

A = np.empty((N,N), dtype=np.double)
b = np.ones(N, dtype=np.double) * 2 * N
x = np.zeros(N, dtype=np.double)

if rank == 0:
    it = np.nditer(A, flags=['multi_index'], op_flags=['writeonly'])
    while not it.finished:
      if it.multi_index[0]==it.multi_index[1]:
          it[0] = N + 1
      else:
          it[0] = 1
      it.iternext()

    comm.Send(A, dest=1,   tag=13)

elif rank == 1:
    comm.Recv(A, source=0, tag=13)


epsilon     = 1e-3
maxit       = 2 * N * N
div_size    = int(N/size)
global_sum  = np.ones(1) * 100
local_sum   = np.zeros(1)

displs      = np.empty(size, dtype=np.double)
recv_counts = np.empty(size, dtype=np.double)
y           = np.empty(div_size, dtype=np.double)


for i in range(0, size):
    displs[i]=i*div_size
    recv_counts[i]=1

for i in range(rank * div_size, (rank + 1) * div_size):
    y[(i-rank*div_size)] = x[i]

for k in range(0, maxit):
    local_sum[0]=0.0
    for i in range(rank * div_size, (rank + 1) * div_size):
        y_new = b[i]
        for j in range(0, N):
            if i != j:
              y_new -= A[i,j] * x[j]

        y_new /= A[i,i]
        dx = y[i-rank*div_size]-y_new
        y[i-rank*div_size] = y_new

        comm.Allgatherv([y,div_size,MPI.DOUBLE], [x,div_size,MPI.DOUBLE])

        local_sum[0] += abs(dx)

    comm.Allreduce([local_sum, MPI.DOUBLE], [global_sum, MPI.DOUBLE], op=MPI.SUM)

    if global_sum[0] <= epsilon:
        break

if rank==0:
    print('residual %f reached after %d iterations\n' % (global_sum[0], k))
    print('A . x = ', np.dot(A,x))
    print("b = ", b)
