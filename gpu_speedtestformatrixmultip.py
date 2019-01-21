import numpy as np
from pycuda import compiler, gpuarray, tools
import pycuda.driver as drv


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






MATRIX_SIZE_vec = np.arange(100,1100,100).tolist()
BLOCK_SIZE = 32
rep = 1

start = drv.Event()
end = drv.Event()

cpu_time = []
gpu_time = []

for matrixsize in MATRIX_SIZE_vec:
    # # create a random vector
    # create two random square matrices
    a_cpu = np.random.randn(matrixsize, matrixsize).astype(np.float32)
    b_cpu = np.random.randn(matrixsize, matrixsize).astype(np.float32)

    # compute reference on the CPU to verify GPU computation


    start.record() # start timing
    start.synchronize()
    for i in range(rep):
        c_cpu = np.dot(a_cpu, b_cpu)
    end.record() # end timing
    # calculate the run length
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print("CPU time:")
    print("%fs" % (secs))
    cpu_time.append(secs)

    # transfer host (CPU) memory to device (GPU) memory
    a_gpu = gpuarray.to_gpu(a_cpu)
    b_gpu = gpuarray.to_gpu(b_cpu)

    # create empty gpu array for the result (C = A * B)
    c_gpu = gpuarray.empty((matrixsize, matrixsize), np.float32)

    start.record()
    for i in range(rep):
        # transfer host (CPU) memory to device (GPU) memory

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