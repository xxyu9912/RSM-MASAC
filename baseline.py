# import the experiment-specific parameters from flow.benchmarks
from flow.benchmarks.figureeight9 import flow_params

# import the make_create_env to register the environment with OpenAI gym
from flow.utils.registry import make_create_env


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch.distributions import Normal


def main():
    # the make_create_env function produces a method that can be used to
    # generate parameterizable gym environments that are compatible with Flow.
    # This method will run both "register" and "make" (see gym documentation).
    # If these are supposed to be run within your algorithm/library, we
    # recommend referring to the make_create_env source code in
    # flow/utils/registry.py.
    create_env, env_name = make_create_env(flow_params, version=0)

    # create and register the environment with OpenAI Gym
    env = create_env()

    # setup the algorithm with the traffic environment. This may be either by
    # specifying the environment's name or the env variable created by the
    # create_env() method
    # alg = myAlgorithm(env_name=env_name)
    # alg.train()

    # recording data
    # hyper parameters
    horizon = 1500
    num_steps = 250
    num_updates = 310

    score = 0.0
    training_score = []
    training_interval = []
    print_interval = 10

    # begin simulation
    for n_step in range(num_updates):
        # initialize environment
        obs = env.reset()
        done = False
        # epoch
        while not done:
            # forward step
            for step in range(num_steps):
                obs_prime, reward, done, _ = env.step(None)
                obs = obs_prime
                score += reward / horizon
                if done:
                    break

        # recording
        if (n_step+1) % print_interval == 0 :
            print('# of episode: {}, avg score: {}'.format(n_step, score / print_interval))
            training_score.append(score / print_interval)  # average
            training_interval.append(n_step)
            score = 0.0

    # plot
    plt.figure(1)
    plt.plot(training_interval, training_score, 'b.-', linewidth=0.5)
    plt.title('Training Performance')
    plt.xlabel('Epochs')
    plt.ylabel('Normalized Average Speed')
    plt.axis([0, 300, 0, 1])
    plt.show()

    np.save('./baseline.npy', training_score)


if __name__ == "__main__":
    main()
