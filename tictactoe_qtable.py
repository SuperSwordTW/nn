from enum import Enum
import os
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import numpy as np
import tf_agents
from tensorflow import keras
from tf_agents.environments import suite_gym
from collections import deque

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import policy_step
from tf_agents.environments.utils import validate_py_environment
import matplotlib.pyplot as plt
import pandas as pd
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy
from tf_agents.environments import tf_py_environment

import random
import collections
import time
import pickle
import sys
import io
import gym
from gym import spaces

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def state_to_index(state):
    arr = state.flatten().astype(int)
    return int(''.join(map(str,arr)), 3)

class TicTacToeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Box(0, 2, (3,3), int)
        self.reset()

    def reset(self):
        self.board = np.zeros((3,3), int)
        self.current_player = 1
        return self.board.copy()

    def step(self, action):
        r, c = divmod(action, 3)
        if self.board[r, c] != 0:
            return self.board.copy(), -1, True, {}
        self.board[r, c] = self.current_player
        if self._win(self.current_player):
            return self.board.copy(), 1, True, {}
        if np.all(self.board != 0):
            return self.board.copy(), 0, True, {}
        self.current_player = 3 - self.current_player
        return self.board.copy(), -0.01, False, {}

    def render(self, mode='human'):
        print(self.board)

    def _win(self, p):
        b = self.board
        # rows, cols
        for i in range(3):
            if np.all(b[i,:]==p) or np.all(b[:,i]==p):
                return True
        # diags
        if b[0,0]==b[1,1]==b[2,2]==p or b[0,2]==b[1,1]==b[2,0]==p:
            return True
        return False

class RandomAgent:
    def __init__(self):
        pass
    def select(self, observation):
        available_actions = [i for i in range(9) if observation.flatten()[i] == 0]
        return random.choice(available_actions)

class MinimaxAgent:
    def __init__(self): self.cache = {}
    def select(self, b): _, a = self._minimax(b, 2, 2); return a
    def _minimax(self, b, me, player):
        key = tuple(b.flatten())
        if key in self.cache: return self.cache[key]
        # terminal checks
        if self._win(b, me): return (10, None)
        if self._win(b, 3-me): return (-10, None)
        if np.all(b!=0): return (0, None)
        best_a, best_v = None, -np.inf if player==me else np.inf
        for a in [i for i,v in enumerate(b.flatten()) if v==0]:
            b2 = b.copy(); r,c = divmod(a,3); b2[r,c] = player
            v,_ = self._minimax(b2, me, 3-player)
            if player==me and v>best_v or player!=me and v<best_v:
                best_v, best_a = v, a
        self.cache[key] = (best_v, best_a)
        return best_v, best_a
    def _win(self, b, p):
        for i in range(3):
            if np.all(b[i]==p) or np.all(b[:,i]==p): return True
        if b[0,0]==b[1,1]==b[2,2]==p or b[0,2]==b[1,1]==b[2,0]==p:
            return True
        return False

class QAgent:
    def __init__(self, env, lr=0.1, gamma=0.9,
                 eps=1.0, eps_decay=0.99999, eps_min=0.01):
        self.env = env
        self.q = np.zeros((3**9, env.action_space.n))
        self.lr, self.gamma = lr, gamma
        self.eps, self.eps_decay, self.eps_min = eps, eps_decay, eps_min

    def select(self, state):
        idx = state_to_index(state)
        avail = [a for a,v in enumerate(state.flatten()) if v==0]
        if random.random() < self.eps:
            return random.choice(avail)
        qs = self.q[idx]
        return max(avail, key=lambda a: qs[a])

    def learn(self, state, action, reward, next_state, done):
        i, j = state_to_index(state), state_to_index(next_state)
        target = reward + (0 if done else self.gamma * np.max(self.q[j]))
        self.q[i,action] += self.lr * (target - self.q[i,action])

    def decay(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

def eval(env, agent, episodes=100):
    wins = 0
    losses = 0
    opponent = MinimaxAgent()
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select(state)
            state, reward, done, _ = env.step(action)
            if reward == 1:
                wins += 1
                break
            if not done:
                action2 = opponent.select(state)
                s2, r2, done, _ = env.step(action2)
                if r2 == 1:
                    losses += 1
                    break
                state = s2
    return wins / episodes, losses / episodes, (episode-wins-losses) / episodes


if __name__=='__main__':
    env = TicTacToeEnv()
    agent = QAgent()
    opponent = RandomAgent()

    episodes = 10000
    recent_outcomes = deque(maxlen=10)

    for ep in range(1, episodes+1):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:

            a = agent.select(state)
            s1, r1, done, _ = env.step(a)

            if not done:
                a2 = opponent.select(s1)
                s2, r2, done, _ = env.step(a2)

                reward = r1 + (-1 if r2==1 else 0)
                next_state = s2
            else:
                reward = r1
                next_state = s1


            agent.store(state, a, reward, next_state, done)
            agent.learn()
            agent.on_step()

            state = next_state
            ep_reward += reward


        agent.decay_eps()
        recent_outcomes.append(1 if ep_reward>0 else 0)


        if ep % 10 == 0:
            win_rate = sum(recent_outcomes)/len(recent_outcomes)
            print(f"Episode {ep}, Epsilon {agent.eps:.3f}, AvgReward {ep_reward:.2f}, WinRate(last10) {win_rate:.2f}")

    print("Training complete.")


    # q.model = keras.models.load_model("tictactoe_model.h5")
    agent.eps = 0


    state = env.reset(); done=False
    while not done:
        env.render()
        if env.current_player==1:
            a = agent.select(state)
            print("Q agent chooses", a)
        else:
            # a = opp.select(state)
            a = int(input("Your move (0-8): "))
        state, r, done, _ = env.step(a)
    env.render()
    if r==1: print(env.current_player, "wins!")
    elif r==-1: print("Invalid move!")
    else: print("Draw!")


    agent.model.save("tictactoe_model.h5")
    print("Win/Lose/Draw",eval(env, agent, 100))