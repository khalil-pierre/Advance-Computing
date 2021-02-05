#------------------Prime Number Counter----------------
from mpi4py import MPI

def prime_number ( n, id, p ):
    total=0
    for i in range(2+id, n+1, p):
        prime = 1
        for j in range(2, i, 1):
            if ( ( i % j ) == 0 ):
                prime = 0
                break
        total += prime
    return total

n_factor=2
n_lo=1
n_hi=65536
primes=0

comm = MPI.COMM_WORLD
p = comm.Get_size()
id = comm.Get_rank()
    
if ( id == 0 ):
    print ( "PRIME_MPI:" )
    print ( "  Python/mpi4py version" )
    print ( "  MPI program to count primes" )
    print ( "  Number of processes is", p )
    print ( "    N   Primes     Time" )

n = n_lo;
while ( n <= n_hi ):
    if ( id == 0 ):
        wtime = MPI.Wtime()
    comm.bcast(n, root=0)
    primes_part = prime_number ( n, id, p )
    primes = comm.reduce (primes_part, None, root=0)
    if ( id == 0 ):
        wtime = MPI.Wtime ( ) - wtime
        print ( "{:5d}  {:5d}    {:10.8f}  ".format(n, primes, wtime) )
    n = n * n_factor
if ( id == 0 ):
    print (" ")
    print ("PRIME_MPI:")
    print ("  Normal end of execution.")

