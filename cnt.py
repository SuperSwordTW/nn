from enum import Enum
import os
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
import json
from keras import backend as K
import gc
import bisect

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def encode_board(board, current_player):
    board = np.array(board).reshape(4, 4, 4)
    encoded = np.zeros((4, 4, 4, 2), dtype=np.float32)
    encoded[board == current_player, 0] = 1.0
    encoded[board == 3 - current_player, 1] = 1.0
    return encoded

def valid_actions(board):
    top_layer = board[:, :, 0]
    return [i for i in range(16) if top_layer[i // 4, i % 4] == 0]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []

    def add(self, transition, td_error=1.0):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            # FIFO replace oldest
            self.buffer.pop(0)
            self.priorities.pop(0)
            self.buffer.append(transition)
            self.priorities.append(priority)

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []

        # Normalize priorities
        scaled = np.array(self.priorities)
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = (abs(td_error) + 1e-6) ** self.alpha

    def clear(self):
        self.buffer.clear()
        self.priorities.clear()

class Score4Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(0, 2, (4,4,4), int)
        self.reset()

    def reset(self):
        self.board = np.zeros((4,4,4), int)
        self.current_player = 1
        return self.board.copy()

    def step(self, action):
        r, c = divmod(action, 4)
        if self.board[r, c, 0] != 0:
            return self.board.copy(), -5.0, True, {}
        i = 0
        while i < 4 and self.board[r, c, i] == 0:
            i += 1
        i-= 1
        self.board[r, c, i] = self.current_player

        result = self._check(r, c, i)
         # Terminal conditions
        if result == 4:
            return self.board.copy(), 10.0, True, {}  # Strong win reward

        if not valid_actions(self.board):
            return self.board.copy(), 0.0, True, {}  # Draw

        reward = 0.0

        # Offensive shaping
        if result == 3:
            reward += 1.0
        elif result == 2:
            reward += 0.05

        # Optional: reward blocking threat
        if self._just_blocked_threat(r, c, i):
            reward += 2.0

        # Defensive shaping
        if self._opponent_has_threat():
            reward -= 0.1

        if self._creates_fork(r, c, i):
            reward += 2.0


        if self._opponent_can_win_next():
            reward -= 4.0

        self.current_player = 3 - self.current_player

        reward -= 0.05  # Slight step penalty

        return self.board.copy(), reward, False, {}

    def render(self, mode='human'):
        print(self.board)

    def _just_blocked_threat(self, x, y, z):
        """
        Returns True if this move blocked an opponent's 3-in-a-row
        by filling an open slot they were building toward.
        """
        # Set the position to 0 temporarily and check for threat
        prev_player = self.current_player
        self.board[x, y, z] = 0
        threat = self._opponent_can_win_next()
        self.board[x, y, z] = prev_player
        return threat
    
    def _creates_fork(self, x, y, z):
        """Returns True if this move creates two or more simultaneous 3-in-a-rows."""
        fork_count = 0
        for dx in range(4):
            for dy in range(4):
                for dz in range(4):
                    if self.board[dx, dy, dz] == self.current_player:
                        threat = self._check(dx, dy, dz)
                        if threat == 3:
                            fork_count += 1
                            if fork_count >= 2:
                                return True
        return False
    
    def _opponent_can_win_next(self):
        opponent = 3 - self.current_player
        for action in range(16):
            r, c = divmod(action, 4)
            if self.board[r, c, 0] != 0:
                continue

            i = 0
            while i < 4 and self.board[r, c, i] == 0:
                i += 1
            i-= 1
            self.board[r, c, i] = opponent
            if self._check(r,c,i) == 4:
                self.board[r, c, i] = 0
                return True
            self.board[r, c, i] = 0
        return False

    def _check(self, x, y, z):
        def check_line(start, direction):
            count = 1  # Start with the current position
            space = 0
            for sign in [-1, 1]:  # Check in both positive and negative directions
                for step in range(1, 4):  # Check up to 3 steps in the given direction
                    nx, ny, nz = start[0] + sign * step * direction[0], start[1] + sign * step * direction[1], start[2] + sign * step * direction[2]
                    if 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and self.board[nx][ny][nz] == self.board[start[0]][start[1]][start[2]] and self.board[nx][ny][nz] != 0:
                        count += 1
                    elif 0 <= nx < 4 and 0 <= ny < 4 and 0 <= nz < 4 and self.board[nx][ny][nz] == 0:
                        space += 1
                    else:
                        break
            return count, space

        directions = [
            [(0, 0, 1)],  # vertical
            [(1, 0, 0), (-1, 0, 0)],  # horizontal x
            [(0, 1, 0), (0, -1, 0)],  # horizontal y
            [(1, 1, 0), (-1, -1, 0)],  # diagonal xy
            [(1, -1, 0), (-1, 1, 0)],  # diagonal xy
            [(1, 0, 1), (-1, 0, -1)],  # diagonal xz
            [(0, 1, 1), (0, -1, -1)],  # diagonal yz
            [(1, 1, 1), (-1, -1, -1)],  # diagonal xyz
            [(1, -1, 1), (-1, 1, -1)],  # diagonal xyz
        ]
        opening = 0
        for direction in directions:
            for d in direction:
                cnt, sp = check_line((x, y, z), d)
                if cnt >= 4:
                    return 4 # win
                elif sp == 1 and cnt == 3:
                    opening = max(opening, cnt)
                elif sp == 2 and cnt == 2:
                    opening = max(opening, cnt)
        return opening
    
    def _opponent_has_threat(self):
        opponent = 3 - self.current_player
        for x in range(4):
            for y in range(4):
                for z in range(4):
                    if self.board[x, y, z] == opponent:
                        if self._check(x, y, z) == 3:
                            return True
        return False

class RandomAgent:
    def __init__(self):
        pass
    def select(self, env):
        observation = env.board
        available_actions = valid_actions(observation)
        if not available_actions:
            return 0
        return random.choice(available_actions)
    
class GreedyAgent:
    def __init__(self):
        self.slot = -1
        pass
    def select(self, env):
        observation = env.board
        available_actions = valid_actions(observation)
        if not available_actions:
            return 0
        flag = 0
        while not(self.slot in available_actions) or flag == 0:
            self.slot += 1
            self.slot %= 16
            flag = 1
        return self.slot

class MinimaxAgent:
    def __init__(self, depth=2, id=1):
        self.depth = depth
        self.player_id = id
        self.lines = self._generate_lines()

    def select(self, env):
        _, action = self._minimax(env, self.depth, maximizing=True, alpha=-np.inf, beta=np.inf)
        return action

    def _minimax(self, env, depth, maximizing, alpha, beta):
        legal_actions = self._available_actions(env)
        if depth == 0 or len(legal_actions) == 0:
            return self._evaluate(env.board), None

        best_action = None
        if maximizing:
            max_eval = -np.inf
            for action in legal_actions:
                next_env, last_move = self._simulate_env(env, action)
                if next_env is None:
                    continue
                if next_env._check(*last_move) == 4:
                    return 100 * (depth + 1), action
                eval_score, _ = self._minimax(next_env, depth - 1, False, alpha, beta)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_action = action
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval, best_action
        else:
            min_eval = np.inf
            for action in legal_actions:
                next_env, last_move = self._simulate_env(env, action)
                if next_env is None:
                    continue
                if next_env._check(*last_move) == 4:
                    return -100 * (depth + 1), action
                eval_score, _ = self._minimax(next_env, depth - 1, True, alpha, beta)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_action = action
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval, best_action

    def _available_actions(self, env):
        return [i for i in range(16) if env.board[i // 4, i % 4, 0] == 0]

    def _simulate_env(self, env, action):
        sim_env = Score4Env()
        sim_env.board = env.board.copy()
        sim_env.current_player = env.current_player
        r, c = divmod(action, 4)
        if sim_env.board[r, c, 0] != 0:
            return None, None
        i = 0
        while i < 4 and sim_env.board[r, c, i] == 0:
            i += 1
        i -= 1
        sim_env.board[r, c, i] = sim_env.current_player
        last_move = (r, c, i)
        sim_env.current_player = 3 - sim_env.current_player
        return sim_env, last_move

    def _evaluate(self, board):
        score = 0
        for line in self.lines:
            values = [board[x, y, z] for x, y, z in line]
            score += self._evaluate_line(values, self.player_id)
        return score

    def _evaluate_line(self, values, agent_id):
        p1 = values.count(agent_id)
        p2 = values.count(3 - agent_id)

        if p1 > 0 and p2 > 0:
            return 0

        if p1 == 4:
            return 1000
        if p2 == 4:
            return -1000
        if p1 == 3 and p2 == 0:
            return 50
        if p2 == 3 and p1 == 0:
            return -50
        if p1 == 2 and p2 == 0:
            return 10
        if p2 == 2 and p1 == 0:
            return -10
        if p1 == 1 and p2 == 0:
            return 1
        if p2 == 1 and p1 == 0:
            return -1
        return 0


    def _generate_lines(self):
        lines = []

        # Straight lines in each dimension
        for x in range(4):
            for y in range(4):
                lines.append([(x, y, z) for z in range(4)])  # vertical
                lines.append([(x, z, y) for z in range(4)])  # x-row in y-z plane
                lines.append([(z, x, y) for z in range(4)])  # y-row in x-z plane

        # 2D diagonals in layers
        for z in range(4):
            lines.append([(i, i, z) for i in range(4)])
            lines.append([(i, 3 - i, z) for i in range(4)])
        for x in range(4):
            lines.append([(x, i, i) for i in range(4)])
            lines.append([(x, i, 3 - i) for i in range(4)])
        for y in range(4):
            lines.append([(i, y, i) for i in range(4)])
            lines.append([(i, y, 3 - i) for i in range(4)])

        # Main space diagonals
        lines.append([(i, i, i) for i in range(4)])
        lines.append([(i, i, 3 - i) for i in range(4)])
        lines.append([(i, 3 - i, i) for i in range(4)])
        lines.append([(3 - i, i, i) for i in range(4)])

        return lines
    
class NoisyMinimaxAgent(MinimaxAgent):
    def __init__(self, depth=1, id=2, noise_prob=1.0):
        super().__init__(depth, id=id)
        self.noise_prob = noise_prob

    def select(self, env):
        if random.random() < self.noise_prob:
            return random.choice(valid_actions(env.board))
        return super().select(env)

class DQNAgent:
    def __init__(self, env, lr=0.00005, gamma=0.99,
                 eps=1.0, eps_decay=0.995, eps_min=0.01,
                 buffer_size=20000, batch_size=128,
                 target_update_freq=500):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        # Build networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network(hard=True)
        self.train_step = 0

    def _build_model(self):
        input_layer = keras.layers.Input(shape=(4,4,4,2))
        x = keras.layers.Conv3D(128, kernel_size=2, activation='relu')(input_layer)
        x = keras.layers.Conv3D(256, kernel_size=2, activation='relu')(x)
        x = keras.layers.Flatten()(x)

        # Dueling streams
        value = keras.layers.Dense(128, activation='relu')(x)
        value = keras.layers.Dense(1)(value)

        advantage = keras.layers.Dense(128, activation='relu')(x)
        advantage = keras.layers.Dense(self.env.action_space.n)(advantage)

        # Combine
        q_values = keras.layers.Lambda(
            lambda inputs: inputs[0] + (inputs[1] - tf.reduce_mean(inputs[1], axis=1, keepdims=True))
        )([value, advantage])

        model = keras.Model(inputs=input_layer, outputs=q_values)
        model.compile(optimizer=keras.optimizers.Adam(self.lr), loss='huber')
        return model

    def update_target_network(self, hard=False):
        if hard:
            self.target_model.set_weights(self.model.get_weights())
        else:
            # soft update
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()
            for i in range(len(weights)):
                target_weights[i] = 0.1 * weights[i] + 0.9 * target_weights[i]
            self.target_model.set_weights(target_weights)

    def select(self, state):
        avail = valid_actions(state)
        if not avail:
            return 0  # fallback to action 0 if everything is full

        if np.random.rand() < self.eps:
            return random.choice(avail)

        inp = encode_board(state, self.env.current_player)
        inp = inp[np.newaxis, ...]
        q_vals = self.model.predict(inp, verbose=0)[0]

        # Mask out invalid actions
        masked_q_vals = [q_vals[a] if a in avail else -np.inf for a in range(16)]
        return int(np.argmax(masked_q_vals))

    def store(self, state, action, reward, next_state, done, current_player):
        self.replay_buffer.add((state, action, reward, next_state, done, current_player))

    def learn(self):
        if len(self.replay_buffer.buffer) < self.batch_size:
            return None  # Not enough data

        batch, indices = self.replay_buffer.sample(self.batch_size)
        states = np.array([encode_board(s, p) for s,a,r,ns,d,p in batch])
        next_states = np.array([encode_board(ns, p) for s,a,r,ns,d,p in batch])
        actions = np.array([a for s,a,r,ns,d,p in batch])
        rewards = np.array([r for s,a,r,ns,d,p in batch])
        dones = np.array([d for s,a,r,ns,d,p in batch])

        q_current = self.model.predict(states, verbose=0)
        q_next_target = q_next_eval = self.target_model.predict(next_states, verbose=0)

        # Mask invalid actions in next states
        for i in range(self.batch_size):
            state_flat = next_states[i, ..., 0] + next_states[i, ..., 1]
            flat_board = state_flat.reshape(4, 4, 4)
            top_layer = flat_board[:, :, 0]
            mask = np.array([1 if top_layer[r, c] == 0 else 0 for r in range(4) for c in range(4)])
            q_next_eval[i] = [q if mask[j] == 1 else -np.inf for j, q in enumerate(q_next_eval[i])]
            q_next_target[i] = [q if mask[j] == 1 else 0.0 for j, q in enumerate(q_next_target[i])]

        next_actions = np.argmax(q_next_eval, axis=1)
        target_q_values = rewards + self.gamma * q_next_target[np.arange(self.batch_size), next_actions] * (1 - dones)

        td_errors = target_q_values - q_current[np.arange(self.batch_size), actions]
        q_current[np.arange(self.batch_size), actions] = target_q_values

        # Mask invalid Q-values in current state
        for i in range(self.batch_size):
            state_flat = states[i, ..., 0] + states[i, ..., 1]
            flat_board = state_flat.reshape(4, 4, 4)
            top_layer = flat_board[:, :, 0]
            invalid_mask = np.array([0 if top_layer[r, c] == 0 else 1 for r in range(4) for c in range(4)])
            q_current[i][invalid_mask == 1] = 0

        # 🔍 Q-value diagnostics (before training)
        q_vals_flat = q_current[np.arange(self.batch_size), actions]
        q_mean = np.mean(q_vals_flat)
        q_max = np.max(q_vals_flat)
        q_min = np.min(q_vals_flat)

        history = self.model.fit(states, q_current, epochs=1, verbose=0)
        self.replay_buffer.update_priorities(indices, td_errors)

        loss = history.history['loss'][0]
        return loss, q_mean, q_max, q_min

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def on_step(self):
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()

class PrevAgent:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def select(self, env):
        inp = encode_board(env.board, env.current_player)
        inp = inp[np.newaxis, ...]
        q_vals = self.model.predict(inp, verbose=0)[0]

        avail = valid_actions(env.board)
        if not avail:
            return 0

        # Mask invalid actions
        masked_q_vals = [q_vals[a] if a in avail else -np.inf for a in range(16)]
        return int(np.argmax(masked_q_vals))
    
class SnapshotAgent:
    def __init__(self, model):
        self.model = model

    def select(self, env):
        inp = encode_board(env.board, env.current_player)
        inp = inp[np.newaxis, ...]
        q_vals = self.model.predict(inp, verbose=0)[0]

        avail = valid_actions(env.board)
        if not avail:
            return 0

        # Mask invalid actions
        masked_q_vals = [q_vals[a] if a in avail else -np.inf for a in range(16)]
        return int(np.argmax(masked_q_vals))
    

def eval(env, agent,opp,episodes=10):
    wins = 0
    losses = 0
    illegal_moves = 0
    opponent = opp
    same = 0
    moves = 0
    ep_reward = 0
    # supervise_agent = MinimaxAgent(depth=3, id=1)
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select(state)
            # supervise_action = supervise_agent.select(env)
            state, reward, done, _ = env.step(action)
            # if action == supervise_action:
            #     same+=1
            ep_reward += reward
            # print("Reward:", reward)
            moves += 1
            if reward == 10.0:
                wins += 1
                break
            if reward == -5:
                illegal_moves += 1
                losses += 1
            if not done:  # Check if the game is not over before opponent moves
                action2 = opponent.select(env)
                s2, r2, done, _ = env.step(action2)
                if r2 == 10.0:
                    ep_reward -= 10.0
                    losses += 1
                    break
                state = s2
    return wins / episodes, losses / episodes, (episode-wins-losses) / episodes, illegal_moves / episodes, same / moves, moves/episodes, ep_reward / episodes

if __name__ == '__main__':
    previous = sys.argv[1]
    series = sys.argv[2]
    s = sys.argv[3]
    episodes = int(sys.argv[4])
    ep_now =int(sys.argv[5])
    # print(state)
    # previous = "1"
    # series = "2"
    # s = "1"
    # episodes = 1500

    env = Score4Env()
    agent = DQNAgent(env)

    supervise_agent = MinimaxAgent(depth=4, id=1)

    #? enable next
    if previous != "0":
        agent.model = keras.models.load_model(f"Score4_{previous}.h5")
        agent.target_model = keras.models.load_model(f"score4_target_{previous}.h5")

        with open(f"replay_buffer_{previous}.pkl", "rb") as f:
            agent.replay_buffer = pickle.load(f)

        with open(f"training_state_{previous}.json", "r") as f:
            state = json.load(f)
            agent.eps = state["eps"]
            agent.train_step = state["train_step"]



    if s == "0":
        weights = [0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1]
    elif s == "1":
        weights = [0.1, 0.1, 0.6, 0.3, 0.1, 0.1, 0.1]
    elif s == "2":
        weights = [0.08, 0.08, 0.4, 0.4, 0.2, 0.2, 0.2]
    elif s == "3":
        weights = [0.05, 0.05, 0.3, 0.3, 0.5, 0.5, 0.3]
    elif s >= "4":
        weights = [0.05, 0.05, 0.3, 0.3, 0.4, 0.4, 0.5]

    opponent_pool = [
                RandomAgent(),
                GreedyAgent(),
                NoisyMinimaxAgent(depth=1,id=2, noise_prob=0.3),
                MinimaxAgent(depth=1, id=2),
                NoisyMinimaxAgent(depth=2,id=2, noise_prob=0.3),
                NoisyMinimaxAgent(depth=2,id=2, noise_prob=0.1),
                MinimaxAgent(depth=2, id=2)
            ]
    opponent = random.choices(opponent_pool, weights)[0]
    
    recent_outcomes = deque(maxlen=10)
    recent_rewards = deque(maxlen=10)
    recent_loses = deque(maxlen=10)
    loss_history = []
    q_mean_history = []
    q_max_history = []
    q_min_history = []

    moves = 0

    for ep in range(ep_now,ep_now+episodes+1):
        state = env.reset()
        done = False
        ep_reward = 0
        won = False
        lose = False

        while not done:
            # -------- Agent move --------
            supervise_action = supervise_agent.select(env)
            action = agent.select(state)
            next_state, r1, done, _ = env.step(action)
            moves+=1
            reward = r1
            if r1 == -5.0:
                # Illegal move — store and end
                agent.store(state, action, r1, next_state, True, 1)
                result = agent.learn()
                if result is not None and ep % 5 == 0:
                    loss, q_mean, q_max, q_min = result
                    loss_history.append(loss)
                    q_mean_history.append(q_mean)
                    q_max_history.append(q_max)
                    q_min_history.append(q_min)
                    if ep % 500 == 0:
                        print(f"[DEBUG] Ep {ep} | Loss: {loss:.4f}, Q-mean: {q_mean:.2f}, Q-max: {q_max:.2f}, Q-min: {q_min:.2f}")
                agent.on_step()
                ep_reward += r1
                break

            if done:
                reward = r1  # Agent won or draw
                if reward == 10.0:
                    won = True
                agent.store(state, action, reward, next_state, True, 1)
                result = agent.learn()
                if result is not None and ep % 5 == 0:
                    loss, q_mean, q_max, q_min = result
                    loss_history.append(loss)
                    q_mean_history.append(q_mean)
                    q_max_history.append(q_max)
                    q_min_history.append(q_min)
                    if ep % 500 == 0:
                        print(f"[DEBUG] Ep {ep} | Loss: {loss:.4f}, Q-mean: {q_mean:.2f}, Q-max: {q_max:.2f}, Q-min: {q_min:.2f}")
                agent.on_step()
                ep_reward += reward
                break

            # -------- Opponent move --------
            opp_action = opponent.select(env)
            state_after_opp, r2, done, _ = env.step(opp_action)

            if done:
                if r2 == 10.0:
                    reward -= 10.0
                    lose = True

            if supervise_action == action:
                reward += 0.2
            else:
                reward -= 0.2

            # Agent learns from outcome after opponent move
            agent.store(state, action, reward, next_state, done, 1)
            result = agent.learn()
            if result is not None and ep % 5 == 0:
                loss, q_mean, q_max, q_min = result
                loss_history.append(loss)
                q_mean_history.append(q_mean)
                q_max_history.append(q_max)
                q_min_history.append(q_min)
                if ep % 500 == 0:
                        print(f"[DEBUG] Ep {ep} | Loss: {loss:.4f}, Q-mean: {q_mean:.2f}, Q-max: {q_max:.2f}, Q-min: {q_min:.2f}")
            agent.on_step()

            # Next step: use updated env state
            state = state_after_opp
            ep_reward += reward

        # -------- Post-episode updates --------
        agent.decay_eps()
        recent_outcomes.append(1 if won else 0)
        recent_loses.append(1 if lose else 0)
        recent_rewards.append(ep_reward)

        opponent = random.choices(opponent_pool, weights)[0]

        if ep % 500 == 0:
            K.clear_session()
            gc.collect()
            print("Win/Lose/Draw",eval(env, agent, RandomAgent(), 10))
            print("Win/Lose/Draw",eval(env, agent, NoisyMinimaxAgent(depth=1,id=2, noise_prob=0.3), 10))
            print("Win/Lose/Draw",eval(env, agent, MinimaxAgent(depth=1,id=2), 1))
            print("Win/Lose/Draw",eval(env, agent, NoisyMinimaxAgent(depth=2,id=2, noise_prob=0.3), 10))
            print("Win/Lose/Draw",eval(env, agent, MinimaxAgent(depth=2,id=2), 1))

        # Logging
        if ep % 10 == 0:
            win_rate = sum(recent_outcomes)/len(recent_outcomes)
            avg_reward = sum(recent_rewards)/len(recent_rewards)
            if isinstance(opponent, NoisyMinimaxAgent):
                print(f"Episode {ep}, Epsilon {agent.eps:.3f}, Noise {opponent.noise_prob:.3f}, AvgReward {avg_reward:.2f}, WinRate(last10) {win_rate:.2f}, LoseRate(last10) {sum(recent_loses)/len(recent_loses):.2f}, Moves {moves/10.0:.2f}")
            else:
                print(f"Episode {ep}, Epsilon {agent.eps:.3f}, AvgReward {avg_reward:.2f}, WinRate(last10) {win_rate:.2f}, LoseRate(last10) {sum(recent_loses)/len(recent_loses):.2f}, Moves {moves/10.0:.2f}")
            moves = 0

    print("Training complete.")

    # plot graph
    plt.plot(loss_history)
    plt.title("Loss History")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    avg_loss = np.convolve(loss_history, np.ones(50)/50, mode='valid')
    plt.plot(avg_loss)
    plt.savefig(f"loss_history_{series}.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(q_mean_history, label="Q-Mean", color='blue')
    plt.plot(q_max_history, label="Q-Max", color='green')
    plt.plot(q_min_history, label="Q-Min", color='red')
    plt.title("Q-Value Statistics Over Time")
    plt.xlabel("Training Iterations (x5 episodes)")
    plt.ylabel("Q-Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"q_value_stats_{series}.png")
    plt.close()

    with open(f"loss_history_total_{series}.pkl", "wb") as f:
        pickle.dump(loss_history, f)

    agent.model.save(f"Score4_{series}.h5")
    agent.target_model.save(f"score4_target_{series}.h5")

    with open(f"replay_buffer_{series}.pkl", "wb") as f:
        pickle.dump(agent.replay_buffer, f)

    training_state = {
        "eps": agent.eps,
        "train_step": agent.train_step
    }
    with open(f"training_state_{series}.json", "w") as f:
        json.dump(training_state, f)

    

    # agent.model.save("Score4.h5")
    agent.model = keras.models.load_model(f"Score4_{series}.h5")
    agent.eps = 0
    agent.update_target_network(hard=True)
    print(f"Model saved as Score4_{series}.h5")
    print("Win/Lose/Draw",eval(env, agent, RandomAgent(), 50))
    print("Win/Lose/Draw",eval(env, agent, NoisyMinimaxAgent(depth=1,id=2, noise_prob=0.3), 50))
    print("Win/Lose/Draw",eval(env, agent, MinimaxAgent(depth=1,id=2), 10))
    print("Win/Lose/Draw",eval(env, agent, NoisyMinimaxAgent(depth=2,id=2, noise_prob=0.3), 50))
    print("Win/Lose/Draw",eval(env, agent, MinimaxAgent(depth=2,id=2), 10))