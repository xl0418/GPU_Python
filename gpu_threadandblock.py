import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import numpy as np
mod = SourceModule("""
    #include <stdio.h>

    __global__ void say_hi(int matrixsize,float *a, float *c)
    {
      // 2D Thread ID 
    int tx = blockDim.x*blockIdx.x + threadIdx.x; // Compute row index
    int ty = blockDim.y*blockIdx.y + threadIdx.y; // Compute column index

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    if((ty <matrixsize) && (tx < matrixsize))
    {
    float Aelement = a[ty];
    float Belement = a[tx];
    Pvalue = Aelement - Belement;

    }
    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * matrixsize + tx] = Pvalue;
    printf("I am %dth thread in threadIdx.x:%d.threadIdx.y:%d  blockIdx.:%d blockIdx.y:%d blockDim.x:%d blockDim.y:%d\\n",(threadIdx.x+threadIdx.y*blockDim.x+(blockIdx.x*blockDim.x*blockDim.y)+(blockIdx.y*blockDim.x*blockDim.y)),threadIdx.x, threadIdx.y,blockIdx.x,blockIdx.y,blockDim.x,blockDim.y);

    }
    """)

MATRIX_SIZE = 6
BLOCK_SIZE = 5

# # create a random vector
a_cpu = np.array([i for i in range(MATRIX_SIZE)]).astype(np.float32)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

func = mod.get_function("say_hi")
func(np.uint32(MATRIX_SIZE),a_gpu,c_gpu,block=(BLOCK_SIZE,BLOCK_SIZE,1),grid=(2,2,1))
c_gpu


