import numpy as np
from pycuda import compiler, gpuarray, tools
import pycuda.driver as drv
import pycuda.autoinit


kernel_code_template = """
__global__ void matrixmulti(int matrixsize,float *a, float *b, float *c)
{

    // 2D Thread ID 
    int tx = blockDim.x*blockIdx.x + threadIdx.x; // Compute column index
    int ty = blockDim.y*blockIdx.y + threadIdx.y; // Compute row index

    // Each thread loads one row of M and one column of N, 
    //   to produce one element of P.
    if((ty <matrixsize) && (tx < matrixsize))
    {
    // Pvalue is used to store the element of the matrix
    // that is computed by the thread
    float Pvalue = 0;
    for(int k=0; k<matrixsize;++k)
    {
    float Aelement = a[ty*matrixsize +k];
    float Belement = b[k*matrixsize +tx];
    Pvalue += Aelement * Belement;
    }
    c[ty * matrixsize + tx] = Pvalue;
    }

}
"""

MATRIX_SIZE = 150
BLOCK_SIZE = 32

# create two random square matrices
a_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.randn(MATRIX_SIZE, MATRIX_SIZE).astype(np.float32)

# compute reference on the CPU to verify GPU computation
c_cpu = np.dot(a_cpu, b_cpu)

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

# get the kernel code from the template
# by specifying the constant MATRIX_SIZE
# kernel_code = kernel_code_template % {
#     'MATRIX_SIZE': MATRIX_SIZE
#     }

# compile the kernel code
mod = compiler.SourceModule(kernel_code_template)

# get the kernel function from the compiled module
matrixmul = mod.get_function("matrixmulti")


# set grid size
if MATRIX_SIZE%BLOCK_SIZE != 0:
    grid=(MATRIX_SIZE//BLOCK_SIZE+1,MATRIX_SIZE//BLOCK_SIZE+1,1)
else:
    grid=(MATRIX_SIZE//BLOCK_SIZE,MATRIX_SIZE//BLOCK_SIZE,1)

matrixsize=MATRIX_SIZE
# call the kernel on the card
matrixmul(np.uint32(matrixsize),
    # inputs
    a_gpu, b_gpu,
    # output
    c_gpu,
    grid=grid,
    block = (BLOCK_SIZE, BLOCK_SIZE, 1),
    )

c_gpu


np.allclose(c_cpu, c_gpu.get())


