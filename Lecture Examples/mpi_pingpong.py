#
# Simple blocking message passing demo.
# A message is passed back and forth between
# a pair of processes.
#
# Run by typing:
# mpirun -np 2 python mpi_pingpong.py
#

from mpi4py import MPI

size = MPI.COMM_WORLD.Get_size()
if size != 2:
    print("Ping-pong only works with 2 ranks")
    exit(1)

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()
#
# Play ping-pong between ranks 0 and 1
#
PING = 0
PONG = 1
MESSAGE_ID = 5 # arbitrary unique choice
counter = 0
for i in range(10):
    # PING sends to PONG and waits to receive a return message
    if rank == PING:
        print("PING sending %d to PONG" % counter)
        comm.send(counter, dest=PONG, tag=MESSAGE_ID)
        counter = comm.recv(source=PONG, tag=MESSAGE_ID)
        print("PING receiving %d from PONG" % counter)
    # PONG waits for PING's message then returns it, incremented
    elif rank == PONG:
        counter = comm.recv(source=PING, tag=MESSAGE_ID)
        print("PONG receiving %d from PING and incrementing" % counter)
        counter += 1
        print("PONG sending %d to PING" % counter)
        comm.send(counter, dest=PING, tag=MESSAGE_ID)
if rank == PING:
    print("Total length of rally: %d times there and back" % counter)
