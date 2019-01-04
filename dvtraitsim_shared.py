import numpy as np


# select time stamp for sorting
def timeKey(elem):
    return elem[0]


class DVTreeData:
    def __init__(self, path, scalar):
        self.path = path
        self.scalar = scalar
        # load tree data
        self.timelist = self._from_txt(path + 'timelist.csv')
        self.timebranch = self._from_txt(path + 'timebranch.csv')
        self.timeend = self._from_txt(path + 'timeend.csv')
        self.traittable = self._from_txt(path + 'traittable.csv')
        self.ltable = self._from_txt(path + 'Ltable.csv')
        # derived data
        self.parent_index = np.absolute(self.ltable[:, 1]).astype(np.int64)
        self.daughter_index = np.absolute(self.ltable[:, 2]).astype(np.int64)
        self.evo_timelist = max(self.timelist[:, 0]) - self.timelist[:, 0]
        self.timebranch = self.timebranch[:, 0].astype(np.int64) - 1
        self.timeend = self.timeend[:, 0].astype(np.int64) - 1
        # evolution time: speciation time
        self.evo_time = max(self.evo_timelist)
        self.speciate_time = self.evo_timelist[self.timebranch]
        self.extinct_time = self.evo_timelist[self.timeend]
        self.extinct_time[self.extinct_time == self.evo_time] = -1
        self.extinct_time = self.extinct_time[self.extinct_time < self.evo_time]
        self.total_species = len(self.speciate_time)
        # create event array: [time, parent, daughter]
        # extinction event if daughter == -1, speciation event otherwise
        self.events = sorted(self._speciation_events() + self._extinction_events(), key=timeKey)
        # prepare simulation events
        self.sim_events = np.array(self.events)
        self.sim_events[:,0] = scalar * self.sim_events[:,0]
        self.sim_events = np.array(self.sim_events).astype(np.int64)
        self.sim_events = np.append(self.sim_events, [[-1,-1,-1]], axis=0)   # guard
        self.sim_evo_time = (scalar * self.evo_time).astype(np.int64)

    # returns trimmed table as numpy.ndarray
    # removes first row and first column
    def _from_txt(self, file):
        tmp = np.genfromtxt(file, delimiter=',', skip_header=1)
        return np.delete(tmp, (0), axis=1)

    # creates list of speciation events [time, parent, daughter] 
    def _speciation_events(self):
        speciation_events = list()
        for sp in range(2, len(self.speciate_time)):
            speciation_events.append([self.speciate_time[sp], self.parent_index[sp] - 1, self.daughter_index[sp]- 1])
        return speciation_events

    # creates list of extinction events [time, specie, -1] 
    def _extinction_events(self):
        extinctind = np.where(self.extinct_time > 0)
        exttime = self.extinct_time[extinctind]
        extinctindlist = [list(i) for i in extinctind][0]
        exttimelist = list(exttime)
        twoc = list(zip(exttimelist, extinctindlist))
        extinction_events = [twoc[i]+(-1,) for i in range(len(twoc))]
        extinction_events = [list(extinction_events[i]) for i in range(len(extinction_events))]
        return extinction_events




# converts named parameters in numpy.array
def DVParam(gamma, a, K, nu, r, theta, Vmax, inittrait, initpop, initpop_sigma, break_on_mu):
    return np.array([gamma, a, K, nu, r, theta, Vmax, inittrait, initpop, initpop_sigma, 1.0 if break_on_mu else 0.0]).astype(float)

