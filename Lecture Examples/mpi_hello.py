#
# Simple hello world demo.
# This code will run on the number of
# tasks requested.
#
# Run by typing:
# mpirun -np 4 python mpi_hello.py
#

from mpi4py import MPI
import time

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()

print("Hello,from process (rank) {:d} of {:d}, running on processor {:s}".format(rank, size, name))
time.sleep(5)


# print(5)
# time.sleep(10)

