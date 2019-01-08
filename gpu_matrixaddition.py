import numpy as np
from pycuda import compiler, gpuarray, tools
import pycuda.driver as drv

# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}
"""

MATRIX_SIZE = 6
BLOCK_SIZE = 6
start = drv.Event()
end = drv.Event()

# # create a random vector
a_cpu = np.random.rand(MATRIX_SIZE,MATRIX_SIZE).astype(np.float32)
b_cpu = np.random.rand(MATRIX_SIZE,MATRIX_SIZE).astype(np.float32)
# compute reference on the CPU to verify GPU computation
start.record() # start timing
start.synchronize()
c_cpu = a_cpu+b_cpu
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("CPU time:")
print("%fs" % (secs))

# transfer host (CPU) memory to device (GPU) memory
a_gpu = gpuarray.to_gpu(a_cpu)
b_gpu = gpuarray.to_gpu(b_cpu)

# create empty gpu array for the result (C = A * B)
c_gpu = gpuarray.empty((MATRIX_SIZE, MATRIX_SIZE), np.float32)

# get the kernel code from the template
# by specifying the constant MATRIX_SIZE
kernel_code = kernel_code_template

# compile the kernel code
mod = compiler.SourceModule(kernel_code)

# get the kernel function from the compiled module
matrixmul = mod.get_function("com_t")

start.record() # start timing

# set grid size
if MATRIX_SIZE%BLOCK_SIZE != 0:
    grid=(MATRIX_SIZE//BLOCK_SIZE+1,MATRIX_SIZE//BLOCK_SIZE+1,1)
else:
    grid=(MATRIX_SIZE//BLOCK_SIZE,MATRIX_SIZE//BLOCK_SIZE,1)

# call the kernel on the card
matrixmul(
    # inputs
    a_gpu,
    b_gpu,
    # output
    c_gpu,
    grid = grid,
    # (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
    block = (BLOCK_SIZE, BLOCK_SIZE, 1),
    )
end.record() # end timing
end.synchronize()
secs = start.time_till(end)*1e-3
print("GPU time:")
print("%fs" % (secs))


# print the results
print("-" * 80)
print("Matrix A (GPU):")
print(a_gpu.get())

# print("-" * 80)
# print("Matrix B (GPU):")
# print(b_gpu.get())

print("-" * 80)
print("Matrix C (GPU):")
print(c_gpu.get())

print("-" * 80)
print("CPU-GPU difference:")
print(c_cpu - c_gpu.get())

np.allclose(c_cpu, c_gpu.get())