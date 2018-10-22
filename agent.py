"""
DDPG agent.
"""

import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    """Meta agent."""

    def __init__(self, models, action_size=2, seed=0, load_file=None,
                 buffer_size=int(1e5),
                 batch_size=64,
                 update_every=2,
                 gamma=0.99,
                 n_agents=2,
                 noise_start=2.0,
                 noise_decay=0.9999):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.n_agents = n_agents
        self.noise = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
        self.agents = [DDPG(0, models[0], load_file=None), DDPG(1, models[1], load_file=None)]
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        all_next_states = all_next_states.reshape(1, -1)  # reshape 2x24 into 1x48 dim vector
        #print((all_states.shape, all_actions.shape, len(all_rewards), all_next_states.shape, len(all_dones)))
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        # Learn every update_every time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, all_states):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            #print(all_states.shape, state.shape)
            action = agent.act(state, noise=self.noise, add_noise=True)
            self.noise *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1) # reshape 2x2 into 1x4 dim vector

    def learn(self, experiences, gamma):
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, _, _ = experiences
            agent_id = torch.tensor([i])
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            #print('state: {}'.format(state.shape))
            action = agent.actor_local(state)
            #print('action: {}'.format(action.shape))
            all_actions.append(action)
        #print('all_actions: {}'.format(all_actions.shape))

        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences, gamma, all_actions)


class DDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, id, model, action_size=2, seed=0, load_file=None,
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 weight_decay=0.0001):
        """
        Params
        ======
            model: model object
            action_size (int): dimension of each action
            seed (int): Random seed
            load_file (str): path of checkpoint file to load
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
        """
        random.seed(seed)

        self.id = id
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.loss_list = []       # track loss across steps

        # Actor Network
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        # Critic Network
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=lr_critic, weight_decay=weight_decay)

        if load_file:
            self.actor_local.load_state_dict(torch.load(load_file + '.actor.pth'))
            self.actor_target.load_state_dict(torch.load(load_file + '.actor.pth'))
            self.critic_local.load_state_dict(torch.load(load_file + '.critic.pth'))
            self.critic_target.load_state_dict(torch.load(load_file + '.critic.pth'))
            print('Loaded: {}'.format(load_file))

        # Noise process
        self.noise = OUNoise(action_size, seed)

    def act(self, state, noise=1.0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        # calculate action values
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample() * noise
        return np.clip(action, -1, 1)


    def reset(self):
        self.noise.reset()


    def learn(self, agent_id, experiences, gamma, all_actions):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id])
        #print('next_states: {}'.format(next_states.shape))
        actions_next = self.actor_target(next_states.reshape(-1, 2, 24)).reshape(-1, 4)
        #print('actions_next: {}'.format(actions_next.shape))
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)  # TODO no_grad?
        #print('q_targets_next: {}'.format(q_targets_next.shape))
        # compute Q targets for current states (y_i)
        q_expected = self.critic_local(states, actions)
        #print('q_expected: {}'.format(q_expected.shape))
        #print(agent_id, actions_next.shape, q_targets_next.shape, q_expected.shape)
        # compute critic loss
        q_targets = rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
        #print('q_targets: {}'.format(q_targets.shape))
        #print('rewards: {}'.format(rewards.index_select(1, agent_id).shape))
        #print('dones: {}'.format(dones.index_select(1, agent_id).shape))
        #print(agent_id, q_expected.shape, q_targets.shape)
        #huber_loss = torch.nn.SmoothL1Loss()
        #critic_loss = huber_loss(q_expected, q_targets.detach())
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        #print(agent_id, critic_loss)
        # minimize loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # compute actor loss
        self.actor_optimizer.zero_grad()
        #print(states.shape)
        #state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
        #print('state: {}'.format(state.shape))
        #actions_pred = self.actor_local(state)
        #print('actions_pred: {}'.format(actions_pred.shape))

        actions_pred = [actions if i == self.id else actions.detach() for i, actions in enumerate(all_actions)]
        #print(actions_pred[1].shape)
        actions_pred = torch.cat(actions_pred, dim=1)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # minimize loss
        #actor_loss.backward(retain_graph=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

        # ---------------------------- update stats ---------------------------- #
        with torch.no_grad():
            self.loss_list.append(critic_loss.item())


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer():
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """
        random.seed(seed)
        np.random.seed(seed)
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        # DEBUG experience sampling
        #print('sampling:')
        #for i, e in enumerate(experiences):
        #    print('------ experience {}:'.format(i))
        #    print(e.state.shape)
        #    print(len(e.action))
        #    print(len(e.reward))
        #    print(e.next_state.shape)
        #    print(len(e.done))
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
