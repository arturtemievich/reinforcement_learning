import os
import sys
import gym
import numpy as np


class CrossEntropy:
    def __init__(self, environment):
        self.environment = environment
        self.n_states = environment.observation_space.n
        self.n_actions = environment.action_space.n

        # Create stochastic policy
        # policy[s, a] = P(take action a | in state s)
        # initialize the policy 'uniformly'
        self.policy = np.ones((self.n_states, self.n_actions)) / self.n_actions

    def generate_session(self, t_max=10 ** 4):
        states, actions = [], []
        total_reward = 0

        state = self.environment.reset()
        for t in range(t_max):
            action = np.random.choice(list(range(self.n_actions)), p=self.policy[state])

            new_state, reward, done, info = self.environment.step(action)
            states.append(state)
            actions.append(action)
            total_reward += reward

            state = new_state
            if done:
                break
        return states, actions, total_reward

# # self.policy into the def __init__
# def initialize_policy(n_states, n_actions):
#     """Create stochastic policy.
#     policy[s, a] = P(take action a | in state s)
#     initialize the policy 'uniformly'
#     """
#     return np.ones((n_states, n_actions)) / n_actions


# def generate_session(environment, policy, t_max=10 ** 4):
#     states, actions = [], []
#     total_reward = 0
#
#     state = environment.reset()
#
#     for t in range(t_max):
#         action = np.random.choice(list(range(n_actions)), p=policy[state])
#
#         new_state, reward, done, info = environment.step(action)
#         states.append(state)
#         actions.append(action)
#         total_reward += reward
#
#         state = new_state
#         if done:
#             break
#     return states, actions, total_reward


def main():
    environment = gym.make("Taxi-v3")
    environment.reset()
    environment.render()

    n_states, n_actions = environment.observation_space.n, environment.action_space.n
    print(f"n_states: {n_states} | n_actions: {n_actions}")
    policy = initialize_policy(n_states, n_actions)
    print(policy)
    print(policy.shape)


if __name__ == "__main__":
    main()
