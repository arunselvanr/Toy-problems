import random
import gym
from keras.optimizers import Adam as adam
import keras.backend as K
count = 0
import numpy as np
np.random.seed(1234)

from collections import deque

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

episodes = 5000

class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory = deque(maxlen=1000)
        self.gamma = .95
        self.epsilon = 1.0 #exploration parameter
        self.epsilon_decay_rate = .9
        self.epsilon_min = .01
        self.learning_rate = .001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, activation='relu', input_dim=self.state_size))
        model.add(Dense(24, activation='relu'))
        #model.add(Dropout(.4))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=adam(lr=self.learning_rate))

        return model

    def add_to_replay(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def pick_action(self):
        if np.random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)
        else :
            Qs = self.model.predict(state)
            return np.argmax(Qs[0])

    def minibatch_training(self):
        minibatch = random.sample(self.replay_memory, mbatch_size)

        for state, action, reward, next_state, done in minibatch:
            Qsa = self.model.predict(state)
            if done:
                Qsa[0][action] = reward
            else :
                Qs_prime = self.model.predict(next_state)
                next_action = np.argmax(Qs_prime[0])
                Qsa[0][action] = reward + self.gamma * Qs_prime[0][next_action]
            self.model.fit(state, Qsa, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min :
            self.epsilon *= self.epsilon_decay_rate



if __name__ == '__main__' :
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size, action_size)
    done = False
    mbatch_size = 40

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for time in range(1000):
            action = agent.pick_action()
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.add_to_replay(state, action, reward, next_state, done)
            state = next_state

            if done:
                print('exploration = %f, time_score = %d, episode_no = %d' %(agent.epsilon, time, e))
                if time == 499:
                    count += 1
                    print(count)
                break

            if len(agent.replay_memory) > mbatch_size:
                agent.minibatch_training()




