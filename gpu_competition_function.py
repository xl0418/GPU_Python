import numpy as np

import time
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.cumath


def competition_functions(a, zi, nj):
    """ competition functions.

    returns beta = Sum_j( exp(-a(zi-zj)^2) * Nj)
            sigma = Sum_j( 2a * (zi-zj) * exp(-a(zi-zj)^2) * Nj)
            sigmaSqr = Sum_j( 4a^2 * (zi-zj)^2 * exp(-a(zi-zj)^2) * Nj)
    """
    T = zi[:, np.newaxis] - zi  # trait-distance matrix (via 'broadcasting')
    t1 = np.exp(-a * T ** 2) * nj
    t2 = (2 * a) * T
    beta = np.sum(t1, axis=1)
    sigma = np.sum(t2 * t1, axis=1)
    sigmasqr = np.sum(t2 ** 2 * t1, axis=1)
    return beta, sigma, sigmasqr


mod = SourceModule("""
__global__ void competition_functions_t(float *dest, float *a, float *zi, float *nj)
{

  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  for(int n = 0; n < n_iter; n++) {
    a[i] = sin(a[i]);
  }
  dest[i] = a[i];
}
""")

competition_functions_beta = mod.get_function("competition_functions_beta")

# create an array of 1s
a = numpy.ones(nbr_values).astype(numpy.float32)
# create a destination array that will receive the result
dest = numpy.zeros_like(a)

gpusin(drv.Out(dest), drv.In(a), numpy.int32(n_iter), grid=(blocks,1), block=(block_size,1,1) )




mod = SourceModule("""
__global__ void competition_functions_beta(float *dest, float *a, int n_iter)
{

  const int i = blockDim.x*blockIdx.x + threadIdx.x;
  for(int n = 0; n < n_iter; n++) {
    a[i] = sin(a[i]);
  }
  dest[i] = a[i];
}
""")

competition_functions_beta = mod.get_function("competition_functions_beta")

# create an array of 1s
a = numpy.ones(nbr_values).astype(numpy.float32)
# create a destination array that will receive the result
dest = numpy.zeros_like(a)

gpusin(drv.Out(dest), drv.In(a), numpy.int32(n_iter), grid=(blocks,1), block=(block_size,1,1) )

