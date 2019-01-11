import numpy as np
from pycuda import compiler, gpuarray, tools
import pycuda.driver as drv

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void com_t(float *a, float *c)
{

    // 2D Thread ID 
    int tx = blockDim.x*blockIdx.x + threadIdx.x; // Compute row index
    int ty = blockDim.y*blockIdx.y + threadIdx.y; // Compute column index

    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    if((ty <%(MATRIX_SIZE)s) && (tx < %(MATRIX_SIZE)s))
    {
    float Aelement = a[ty];
    float Belement = a[tx];
    Pvalue = Aelement - Belement;
    }
    // Write the matrix to device memory;
    // each thread writes one element
    c[ty * %(MATRIX_SIZE)s + tx] = Pvalue;
}
"""

MATRIX_SIZE = 6
BLOCK_SIZE = 5

# # create a random vector
a_cpu = np.array([i for i in range(MATRIX_SIZE)]).astype(np.float32)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

# get the kernel code from the template
# by specifying the constant MATRIX_SIZE
kernel_code = kernel_code_template % {
    'MATRIX_SIZE': MATRIX_SIZE
    }

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
matrixmul = mod.get_function("com_t")

# set grid size
if MATRIX_SIZE%BLOCK_SIZE != 0:
    grid=(MATRIX_SIZE//BLOCK_SIZE+1,MATRIX_SIZE//BLOCK_SIZE+1,1)
else:
    grid=(MATRIX_SIZE//BLOCK_SIZE,MATRIX_SIZE//BLOCK_SIZE,1)

# call the kernel on the card
matrixmul(
    # inputs
    a_gpu,
    # output
    c_gpu,
    grid = grid,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block = (BLOCK_SIZE, BLOCK_SIZE, 1),
    )
c_gpu