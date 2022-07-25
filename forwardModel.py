from globalParameters import GP
from FC import FC
import tensorflow as tf
import os,sys
log_prefix = '[' + os.path.basename(__file__)

class FM:
    def __init__(self):
        self.state_dim, self.act_dim = GP.get_dim_action_state()
        self.num_nets = GP.n_ensemble
        self.layers = []
        self.end_act, self.end_act_name = None, None
        self.optimizer = None
        self.create_neural_network()

    def add(self, layer):
        layer.set_ensemble_size(self.num_nets)
        if len(self.layers) > 0:
            layer.set_input_dim(self.layers[-1].get_output_dim())
        self.layers.append(layer.copy())

    def finalize(self, optimizer, optimizer_args=None, *args, **kwargs):
        optimizer_args = {} if optimizer_args is None else optimizer_args
        self.optimizer = optimizer(**optimizer_args)
        self.layers[-1].set_output_dim(2*self.layers[-1].get_output_dim())
        self.end_act = self.layers[-1].get_activation()
        self.end_act_name = self.layers[-1].get_activation(as_func=False)
        self.layers[-1].unset_activation()

    def create_neural_network(self):
        self.add(FC(200, input_dim=self.state_dim + self.act_dim, activation="swish", weight_decay=0.000025))
        self.add(FC(200, activation="swish", weight_decay=0.00005))
        self.add(FC(200, activation="swish", weight_decay=0.000075))
        self.add(FC(200, activation="swish", weight_decay=0.000075))
        self.add(FC(self.state_dim, weight_decay=0.0001))
        self.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

    def predict(self, sc, A_avai, ts):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno) + '[line-24][predicting s[%d] using s[%d] and a%s] at time %d', (ts, sc.id, str([a.id for a in A_avai]), ts), 'procedure')