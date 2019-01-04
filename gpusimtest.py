import platform
import sys
if platform.system() == 'Windows':
    sys.path.append('C:/Liang/GPU_Python')
elif platform.system() == 'Darwin':
    sys.path.append('/Users/dudupig/Documents/GitHub/Code/Pro2/Python_p2')

import numpy as np
import timeit as time
from dvtraitsim_shared import DVTreeData, DVParam
import dvtraitsim_cpp as dvcpp
from dvtraitsim_py import DVSim
import pycuda.driver as drv
from gpusim import gpusim
no_tree = 1
K = 10e8
nu = 1 / (100 * K)
gamma = 0.001
a = 0.1
dir_path = 'c:/Liang/Googlebox/Research/Project2'
files = dir_path + '/treesim_newexp/example%d/' % no_tree

td = DVTreeData(path=files, scalar=100000)

obs_param = DVParam(gamma=gamma, a=a, K=K, nu=nu, r=1, theta=0, Vmax=1, inittrait=0, initpop=500,
                    initpop_sigma=10.0, break_on_mu=False)

start = drv.Event()
end = drv.Event()

#####################
## gpupy sim version###
#####################
start.record() # start timing
start.synchronize()
simresult = gpusim(td, obs_param)
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("gpupy time:")
print("%fs" % (secs))

#####################
## cpp sim version###
#####################
start.record() # start timing
start.synchronize()
simresult = dvcpp.DVSim(td, obs_param)
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("cpp time:")
print("%fs" % (secs))


#####################
## py sim version###
#####################
start.record() # start timing
start.synchronize()
simresult = DVSim(td, obs_param)
end.record() # end timing
# calculate the run length
end.synchronize()
secs = start.time_till(end)*1e-3
print("py time:")
print("%fs" % (secs))


