# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:26:39 2021

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

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
        raise ValueError('The size of input arrat must be an even number')
    elif N <= 2:
        print(x)
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
        coefficent_vector = np.exp(-2j * np.pi * np.arange(N) / N)
        f = np.concatenate([even_f + coefficent_vector[:int(N / 2)] * odd_f,
                            even_f + coefficent_vector[int(N / 2):] * odd_f])
        
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












  



