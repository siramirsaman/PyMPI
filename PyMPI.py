#!/usr/bin/env python
"""
Python MPI A.x = b
"""

from mpi4py import MPI
import numpy as np

class linear:
    def __init__ (self, comm, N):
        self.N = N
        self.A = np.empty((N,N), dtype=np.double)
        self.b = np.ones(N, dtype=np.double) * 2 * N
        self.x = np.zeros(N, dtype=np.double)

        self.initialize_mpi(comm)

    def initialize_mpi(self, comm):
        if comm.Get_rank() == 0:
            it = np.nditer(self.A, flags=['multi_index'], op_flags=['writeonly'])
            while not it.finished:
              if it.multi_index[0]==it.multi_index[1]:
                  it[0] = self.N + 1
              else:
                  it[0] = 1
              it.iternext()
            comm.Send(self.A, dest=1,   tag=13)
        else:
            comm.Recv(self.A, source=0, tag=13)


class G:
    def __init__ (self, comm, N):
        self.comm = comm
        self.l = linear(comm, N)
        self.div_size    = int(self.l.N/self.comm.Get_size())
        self.global_sum  = np.ones(1) * 100
        self.local_sum   = np.zeros(1)

        self.displs      = np.empty(self.comm.Get_size(), dtype=np.double)
        self.recv_counts = np.empty(self.comm.Get_size(), dtype=np.double)
        self.y           = np.empty(self.div_size, dtype=np.double)

        self.initialize()


    def initialize(self):
        for i in range(0, self.comm.Get_size()):
            self.displs[i]=i*self.div_size
            self.recv_counts[i]=1

        for i in range(self.comm.Get_rank() * self.div_size, (self.comm.Get_rank() + 1) * self.div_size):
            self.y[(i-self.comm.Get_rank()*self.div_size)] = self.l.x[i]

    def iterate(self, epsilon, maxit):
        for k in range(0, maxit):
            self.local_sum[0]=0.0
            for i in range(self.comm.Get_rank() * self.div_size, (self.comm.Get_rank() + 1) * self.div_size):
                y_new = self.l.b[i]
                for j in range(0, self.l.N):
                    if i != j:
                      y_new -= self.l.A[i,j] * self.l.x[j]

                y_new /= self.l.A[i,i]
                dx = self.y[i-self.comm.Get_rank()*self.div_size]-y_new
                self.y[i-self.comm.Get_rank()*self.div_size] = y_new

                comm.Allgatherv([self.y,self.div_size,MPI.DOUBLE], [self.l.x,self.div_size,MPI.DOUBLE])

                self.local_sum[0] += abs(dx)

            comm.Allreduce([self.local_sum, MPI.DOUBLE], [self.global_sum, MPI.DOUBLE], op=MPI.SUM)

            if self.global_sum[0] <= epsilon:
                break


if __name__ == "__main__":
    N = 10
    epsilon     = 1e-3
    maxit       = 2 * N**2

    comm = MPI.COMM_WORLD

    G1 = G(comm, N)

    G1.iterate(epsilon, maxit)

    if G1.comm.Get_rank()==0:
        print('residual %f\n' % (G1.global_sum[0]))
        print('A . x = ', np.dot(G1.l.A,G1.l.x))
        print("b = ", G1.l.b)
