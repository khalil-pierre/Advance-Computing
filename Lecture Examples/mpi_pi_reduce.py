#**********************************************************************
# FILE: mpi_pi_reduce.py
# DESCRIPTION:
#   MPI pi Calculation Example - Python mpi4py version.
#   Collective Communication example:
#   This program calculates pi using a "dartboard" algorithm.  See
#   Fox et al.(1988) Solving Problems on Concurrent Processors, vol.1
#   page 207.  All processes contribute to the calculation, with the
#   master averaging the values for pi. This version uses mpi_reduce to
#   collect results
# AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
#   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
# LAST REVISED: 06/13/13 Blaise Barney
#   Adapted to Python / mpi4py by SH (27/10/18)
#**********************************************************************
from mpi4py import MPI
import numpy as np

DARTS = 50000     # number of throws at dartboard
ROUNDS = 100      # number of times "darts" is iterated
MASTER = 0        # task ID of master task

#**************************** functions ************************************

def dboard (darts):
    """
    Used in pi calculation example codes.
    Throw darts at board.  Done by generating random numbers
    between 0 and 1 and converting them to values for x and y
    coordinates and then testing to see if they "land" in
    the circle."  If so, score is incremented.  After throwing the
    specified number of darts, pi is calculated.  The computed value
    of pi is returned as the value of this function, dboard.
    """
    r = np.random.rand(darts,2)  # generate set of random numbers for x and y   
    score = 0
    for n in range(darts):  # "throw darts at board"
        x_coord = (2.0 * r[n,0]) - 1.0
        y_coord = (2.0 * r[n,1]) - 1.0
        if (x_coord*x_coord + y_coord*y_coord) <= 1.0:
            score += 1  # if dart lands in circle, increment score
    pi = 4.0 * score/darts
    return pi

#**************************** initializations ************************************
#
comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
taskid = comm.Get_rank()
print ("MPI task %d has started..." % taskid)
#
# Set seed for random number generator equal to task ID, to get different
# random numbers for each task.
#
np.random.seed(taskid)
#
#**************************** all tasks ************************************
avepi = 0
for i in range(ROUNDS):
#
# All tasks calculate pi using dartboard algorithm
#
    homepi = dboard(DARTS)
#
# Use MPI_Reduce to sum values of homepi across all tasks
# Master will store the accumulated value in pisum
# homepi is the send buffer
# pisum is the receive buffer (used by the receiving task only)
#
    pisum = comm.reduce(homepi, None, root=MASTER)
#
# Master computes average for this iteration and all iterations
#
    if (taskid == MASTER):
        pi = pisum/numtasks
        avepi = ((avepi * i) + pi)/(i + 1)
        print("   After %8d throws, average value of pi = %10.8f" % (DARTS * (i + 1)*numtasks,avepi))  

if (taskid == MASTER):
    print ("Real value of PI: 3.1415926535897")

