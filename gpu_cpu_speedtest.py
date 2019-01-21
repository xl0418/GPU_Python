import numpy as np
from pycuda import compiler, gpuarray, tools
import pycuda.driver as drv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# -- initialize the device
import pycuda.autoinit

kernel_code_template = """
__global__ void com_t(int matrixsize,float *a, float *c)
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
    c[ty * matrixsize + tx] = Pvalue;
    }
    // Write the matrix to device memory;
    // each thread writes one element

}
"""

# compile the kernel code
mod = compiler.SourceModule(kernel_code_template)

# get the kernel function from the compiled module
matrixmul = mod.get_function("com_t")

MATRIX_SIZE_vec = np.arange(100,15100,100).tolist()
BLOCK_SIZE = 32
rep = 150000

start = drv.Event()
end = drv.Event()

cpu_time = []
gpu_time = []

for matrixsize in MATRIX_SIZE_vec:
    # # create a random vector
    a_cpu = np.array([i for i in range(matrixsize)]).astype(np.float32)

    start.record() # start timing
    start.synchronize()
    for i in range(rep):
        c_cpu = a_cpu[:,np.newaxis] - a_cpu
    end.record() # end timing
    # calculate the run length
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print("CPU time:")
    print("%fs" % (secs))
    cpu_time.append(secs)

    # transfer host (CPU) memory to device (GPU) memory

    start.record()
    for i in range(rep):
        a_gpu = gpuarray.to_gpu(a_cpu)
        c_gpu = gpuarray.empty((matrixsize, matrixsize), np.float32)
        # set grid size
        if matrixsize%BLOCK_SIZE != 0:
            grid=(matrixsize//BLOCK_SIZE+1,matrixsize//BLOCK_SIZE+1,1)
        else:
            grid=(matrixsize//BLOCK_SIZE,matrixsize//BLOCK_SIZE,1)

        # call the kernel on the card
        matrixmul(np.uint32(matrixsize),
            # inputs
            a_gpu,
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
    gpu_time.append(secs)


time = cpu_time+gpu_time
device = np.repeat(['cpu','gpu'],len(MATRIX_SIZE_vec))
dim = np.tile(MATRIX_SIZE_vec,2)
speedtest = {'Time':time,'Device':device,'Dim':dim}
stdf = pd.DataFrame(speedtest)
sns.lineplot(x="Dim", y="Time", hue="Device",style='Device',  data=stdf)
plt.title('Time for 150k replicates of simple matrix algebra')

