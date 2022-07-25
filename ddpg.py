import random

import numpy as np

from globalParameters import GP
from actions import ACT
import os,sys
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
from collections import deque

log_prefix = '[' + os.path.basename(__file__)


def stack_samples(samples):
    array = np.array(samples)
    s_ts = np.stack(array[:,0]).reshape((array.shape[0], -1))
    actions = np.stack(array[:, 1]).reshape((array.shape[0], -1))
    rewards = np.stack(array[:, 2]).reshape((array.shape[0], -1))
    s_ts1 = np.stack(array[:, 3]).reshape((array.shape[0], -1))
    return s_ts, actions, rewards, s_ts1

class DDPG:
    def __init__(self):
        self.sess = GP.sess
        self.epsilon = 0.9
        self.gamma   = 0.99
        self.epsilon_decay = 0.99995
        self.tau     = 0.01
        self.memory  = deque(maxlen=4000)

        self.state_dim, self.act_dim = GP.get_dim_action_state()

        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()
        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.act_dim])
        actor_model_weights    = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(0.0001).apply_gradients(grads)

        self.critic_state_input, self.critic_action_input, self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()
        self.critic_grads = tf.gradients(self.critic_model.output, self.critic_action_input)

        #self.sess.run(tf.global_variables_initializer())

        self.pending_s = None
        self.pending_a = None

    def create_actor_model(self):
        state_input = Input(shape=(self.state_dim,))
        h1 = Dense(300, activation='relu')(state_input)
        h2 = Dense(400, activation='relu')(h1)
        output = Dense(self.act_dim, activation='tanh')(h2)
        model = Model(inputs=state_input, outputs=output)
        adam = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=(self.state_dim,))
        state_h1 = Dense(300, activation='relu')(state_input)
        state_h2 = Dense(400)(state_h1)
        action_input = Input(shape=(self.act_dim,))
        action_h1 = Dense(400)(action_input)
        merged = Concatenate()([state_h2, action_h1])
        merged_h1 = Dense(500, activation='relu')(merged)
        output = Dense(1, activation='linear')(merged_h1)
        model = Model(inputs=[state_input, action_input], outputs=output)
        adam = Adam(lr=0.0001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    def act(self, state):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'state: %s', (str(state)), 'optional')
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[line-24][generate action value to be executed]', None, 'optional')
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.actor_model.predict(state)*2 + np.random.normal()
        return self.actor_model.predict(state)*2

    def remember(self, s, a, r):
        if self.pending_s is not None:
            self.memory.append([self.pending_s, self.pending_a, r, s])
            self.pending_s, self.pending_a = s, a
        else:
            self.pending_s, self.pending_a = s, a

    def train(self):
        batch_size = 256
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        self.samples = samples
        self.train_critic(samples)
        self.train_actor(samples)

    def train_critic(self, samples):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Training critic]',None,'procedure')
        s_ts, actions, rewards, s_ts1 = stack_samples(samples)
        target_actions = self.target_actor_model.predict(s_ts1)
        future_rewards = self.target_critic_model.predict([s_ts, target_actions])
        rewards += self.gamma*future_rewards
        self.critic_model.fit([s_ts, actions], rewards, verbose=0)

    def train_actor(self, samples):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Training actor]',None,'procedure')
        s_ts, _, _, _ = stack_samples(samples)
        predicted_actions = self.actor_model.predict(s_ts)
        grads = self.sess.run(self.critic_grads, feed_dict={
            self.critic_state_input: s_ts,
            self.critic_action_input: predicted_actions
        })[0]
        self.sess.run(self.optimize, feed_dict={
            self.actor_state_input: s_ts,
            self.actor_critic_grad: grads
        })

    def update_target(self):
        self.update_actor_target()
        self.update_critic_target()

    def update_actor_target(self):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Update actor target]',None,'procedure')
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_actor_model.get_weights()
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]* self.tau + actor_target_weights[i]*(1-self.tau)
        self.target_actor_model.set_weights(actor_target_weights)

    def update_critic_target(self):
        GP.LOG(GP.getLogInfo(log_prefix, sys._getframe().f_lineno)+'[Update critic target]',None,'procedure')
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.target_critic_model.get_weights()
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]*self.tau + critic_target_weights[i]*(1-self.tau)
        self.target_critic_model.set_weights(critic_target_weights)

