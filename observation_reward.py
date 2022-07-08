import numpy as np
from globalParameters import GP
class OBSRWD:
    def __init__(self,id):
        self.id = id
        self.obs_delay = np.random.uniform(1,5,None)
        self.n_ts = self.obs_delay/GP.delta_t