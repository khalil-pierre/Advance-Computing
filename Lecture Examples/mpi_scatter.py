#
# mpi_scatter.py
#
# Simple demonstration of scatter method.
#
# Run by typing:
# mpirun -np 4 python mpi_scatter.py
#

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numtasks = comm.Get_size()

SIZE = numtasks
sendbuf=np.zeros((SIZE,SIZE))
# recvbuf=np.zeros(SIZE)

if numtasks==SIZE:
    # define source task and elements to send/receive,
    # then perform collective scatter
    source = 0
    # sendcount = SIZE
    # recvcount = SIZE
    # only the source task initialises sendbuf
    if rank==source:
        for i in range(SIZE):
            for j in range(SIZE):
                sendbuf[i,j] = i*SIZE + j+1
    # all ranks issue MPI_Scatter command
    recvbuf=comm.scatter(sendbuf,root=source)

    print("rank= {:2d} Results: {}".format(rank,recvbuf))
else:
    print("Must specify",SIZE," processors. Terminating.")
