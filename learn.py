#!/usr/bin/env python

import argparse
import environment, agent, model, training

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="path to saved model", type=str, default=None)
args = parser.parse_args()

# create objects and run training
environment = environment.UnityMLVectorMultiAgent()
n_agents = 2
models = [model.LowDim2x(n_agents=n_agents) for _ in range(n_agents)]
agent = agent.MADDPG(models, n_agents=n_agents, load_file=args.load)
training.train(environment, agent)
