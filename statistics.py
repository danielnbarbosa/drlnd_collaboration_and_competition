"""
Statistics to track agent performance.
"""


import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from visualize import sub_plot
from tensorboardX import SummaryWriter


class Stats():
    def __init__(self):
        self.score = None
        self.avg_score = None
        self.std_dev = None
        self.scores = []                         # list containing scores from each episode
        self.avg_scores = []                     # list containing average scores after each episode
        self.scores_window = deque(maxlen=100)   # last 100 scores
        self.best_avg_score = -np.Inf            # best score for a single episode
        self.time_start = time.time()            # track cumulative wall time
        self.total_steps = 0                     # track cumulative steps taken
        self.writer = SummaryWriter()

    def update(self, steps, rewards, i_episode):
        """Update stats after each episode."""
        self.total_steps += steps
        self.score = sum(rewards)
        self.scores_window.append(self.score)
        self.scores.append(self.score)
        self.avg_score = np.mean(self.scores_window)
        self.avg_scores.append(self.avg_score)
        self.std_dev = np.std(self.scores_window)
        # update best average score
        if self.avg_score > self.best_avg_score and i_episode > 100:
            self.best_avg_score = self.avg_score

    def is_solved(self, i_episode, solve_score):
        """Define solve criteria."""
        return self.avg_score >= solve_score and i_episode >= 100

    def print_episode(self, i_episode, steps, stats_format, buffer_len, noise):
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Steps: {:8}   Reward: {:8.3f}  |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, steps, self.score)
        print( '\r' + common_stats + stats_format.format(buffer_len, noise), end="")
        self.writer.add_scalar('data/reward', self.score, i_episode)
        self.writer.add_scalar('data/std_dev', self.std_dev, i_episode)
        self.writer.add_scalar('data/avg_reward', self.avg_score, i_episode)
        self.writer.add_scalar('data/buffer_len', buffer_len, i_episode)
        self.writer.add_scalar('data/noise', noise, i_episode)


    def print_epoch(self, i_episode, stats_format, *args):
        n_secs = int(time.time() - self.time_start)
        common_stats = 'Episode: {:5}   Avg: {:8.3f}   BestAvg: {:8.3f}   σ: {:8.3f}  |  Steps: {:8}   Secs: {:6}      |  '.format(i_episode, self.avg_score, self.best_avg_score, self.std_dev, self.total_steps, n_secs)
        print('\r' + common_stats + stats_format.format(*args))

    def print_solve(self, i_episode, stats_format, *args):
        self.print_epoch(i_episode, stats_format, *args)
        print('\nSolved in {:d} episodes!'.format(i_episode-100))

    def plot(self, loss_list):
        """Plot stats in nice graphs."""
        window_size = len(loss_list) // 100 # window size is 1% of total steps
        plt.figure(1)
        # plot score
        sub_plot(221, self.scores, y_label='Score')
        sub_plot(223, self.avg_scores, y_label='Avg Score', x_label='Episodes')
        # plot loss
        sub_plot(222, loss_list, y_label='Loss')
        avg_loss = np.convolve(loss_list, np.ones((window_size,))/window_size, mode='valid')
        sub_plot(224, avg_loss, y_label='Avg Loss', x_label='Steps')
        plt.show()
