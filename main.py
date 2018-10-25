#!/usr/bin/env python

"""
Main program.
"""

import argparse
import environment
import agent
import training

parser = argparse.ArgumentParser()
parser.add_argument("--load", help="path to saved model", type=str, default=None)
args = parser.parse_args()

# create environment and agent, then run training
environment = environment.UnityMLVectorMultiAgent()
agent = agent.MADDPG(load_file=args.load)
training.train(environment, agent)
