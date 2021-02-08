# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:26:39 2021

@author: user
"""
#Stops numpy from trying to capture multiple cores on cluster nodes
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def DFT(x,ortho=False):
    '''
    This function will calculate the discrete fourier transform of a 1d array 
    using matrix multiplication. The elements of the transformation matrix are
    given by M_mn = exp(-2j*pi*n*m*N^{-1}), where n and m are the coloumn and 
    row indices respectivly and N is the size of the array.

    Parameters
    ----------
    x : np.darray
        The array to be transformed.
    ortho : bool
        The transformed array will be normalised

    Returns
    -------
    f : np.darray
        Transformed array

    '''
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N) #generate row indices
    m = n.reshape((N, 1)) #generate column indices
    M = np.exp(-2j * np.pi *m * n / N) #generate transformation matrix
    
    if ortho == True:
        norm = 1 / np.sqrt(N)
    else:
        norm = 1
    
    f = norm*np.dot(M, x)
    
    return  f

def FFT(x):
    '''
    This function calculates the DFT of a 1D array using the recursive 1D 
    Cooley-Tukey method.

    Parameters
    ----------
    x : np.darray
        The array to be transformed.

    Returns
    -------
    f : np.darray
        Transformed array

    '''
    
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 >0:
        raise ValueError('The size of input array must be an even number the '
                         + 'size of the input is N = {}'.format(N))
    elif N <= 32:
        n = np.arange(N) #generate row indices
        m = n.reshape((N, 1)) #generate column indices
        M = np.exp(-2j * np.pi *m * n / N) #generate transformation matrix
        
        return np.dot(M, x)
    else:
        even_x = x[::2] #Get all the even elements indexed of 
        odd_x = x[1::2] #Get all the odd elements indexed of 
        
        #If the split arrays have a size of less then 32 then the DFT of the 
        #split arrays will be returned. If not the split arrays will be passed 
        #back through the function until they meet this condition.
        
        even_f = FFT(even_x) #Calculate the DFT of even indexed elements
        odd_f = FFT(odd_x)   #Calculate the DFT of odd indexed elements 
        twiddle_factor = np.exp(-2j * np.pi * np.arange(N/2) / N) 
        f = np.concatenate([even_f + twiddle_factor * odd_f,
                            even_f - twiddle_factor * odd_f])
        
        return f
        
def test_FT(func):
    '''
    Tests a function that computes the discrete fourier transform of a array. 
    It does this by calculating the fourier transorm of an impulse at
    the origin. The output should be a constant transform domain.
    
    Parameters
    ----------
    func : function
        The function you want to test.

    Returns
    -------
    result : bool
        If true the function is correctly calculating the expected fourier 
        transform then the function will return True.

    '''
    x = np.zeros(8)
    x[0] = 1
    X = func(x)
    
    result = True
    for i in X:
        if i != 1:
            result = False
            break
        
    return result

def FFT_vec(x):
    '''
    Vectorised version of the Cooley-Tukey method.

    Parameters
    ----------
    x : np.darray
        The array to be transformed.

    Returns
    -------
    f : np.darray
        Transformed array

    '''
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if np.log2(N) % 1 > 0:
        raise ValueError('The size of x must be a power of 2, size is {}'.format(N))
    
    x = x.reshape(2,-1)
    M = np.array([[1,1],[1,-1]]) # Fourier transform matrix for 2x1 input
    f = np.dot(M, x) #Calculates Fourier transform for all pairs
    
    while f.shape[0] < N:
        f_even = f[:,:int(f.shape[1]/2)]
        f_odd = f[:,int(f.shape[1]/2):]
        
        twiddle_factor = np.exp(-1j * np.pi * np.arange(f.shape[0])
                        / f.shape[0])[:, None]
        f = np.vstack([f_even + twiddle_factor * f_odd,
                       f_even - twiddle_factor * f_odd])
        
    return f.ravel()


#--------------------------------Parallel FFT Design--------------------------

def bit_reversal(x):
    '''
    This function takes an array where the size of the array can be expressed
    as a power of 2 and performs a bit reversal of the array.
    

    Parameters
    ----------
    x : np.darray
        Input array that you want to perform bit reversal too.


    Returns
    -------
    np.darray
        Returns a bit reverse array.

    '''
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if np.log2(N) % 1 > 0:
        raise ValueError('The size of x must be a power of 2, size is {}'.format(N))
    
    elif N == 2:
        return x
    else:
        x_even = bit_reversal(x[::2])
        x_odd = bit_reversal(x[1::2])
        
        return np.concatenate([x_even,x_odd])
    
def coef_matrix_2x2():
    '''
    This function returns the 2x2 fourier transform matrix. Calculating this 
    using the coefficent of the transform matrix requires extra steps that are
    redundant as the form is simple.

    Returns
    -------
    matrix : np.darray
        The 2x2 fourier transform matrix.

    '''
    matrix = np.array([[1,1],[1,-1]])
    return matrix
    
    
    
# x = reorder_data(x)

#I am gonna Start by writing a bit reversal function 
x = np.arange(16)
N = x.shape[0]
p = np.log2(N)

#This works using my current FFT function

# core_1 = x[::2]
# core_2 = x[1::2]

# core_1_f = FFT(core_1)
# core_2_f = FFT(core_2)

# twiddle_factor = np.exp(-2j * np.pi * np.arange(N/2) / N) 

# f = np.concatenate([core_1_f + twiddle_factor * core_2_f,
#                     core_1_f - twiddle_factor * core_2_f])

# print(f)

# print(np.fft.fft(x))


x = np.asarray(x, dtype = float)
N = x.shape[0]

if np.log2(N) % 1 > 0:
    raise ValueError('The size of x must be a power of 2, size is {}'.format(N))
    
N_min = 2
M = coef_matrix_2x2()

x = x.reshape((2,-1))

f = np.dot(M, x.reshape((N_min, -1)))

f_even = f[:,:int(f.shape[1]/2)]
f_odd = f[:,int(f.shape[1]/2):]

twiddle_factor = np.exp(-1j * np.pi * np.arange(f.shape[0])
                        / f.shape[0])[:, None]

    

