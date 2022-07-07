import os
from copy import deepcopy

import numpy as np
import numpy.random as rd
import torch
from elegantrl.net import Actor
from elegantrl.net import ActorBiConv
from elegantrl.net import ActorDiscretePPO
from elegantrl.net import ActorPPO
from elegantrl.net import ActorSAC
from elegantrl.net import Critic
from elegantrl.net import CriticBiConv
from elegantrl.net import CriticPPO
from elegantrl.net import CriticTwin
from elegantrl.net import QNet
from elegantrl.net import QNetDuel
from elegantrl.net import QNetTwin
from elegantrl.net import QNetTwinDuel
from elegantrl.net import ShareBiConv
from elegantrl.net import ShareDPG
from elegantrl.net import SharePPO
from elegantrl.net import ShareSPG
from torch.nn.utils import clip_grad_norm_

"""[ElegantRL.2021.11.05](https://github.com/AI4Finance-Foundation/ElegantRL)"""


class AgentBase:
    def __init__(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        """initialize

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param reward_scale: scale the reward to get a appropriate scale Q value
        :param gamma: the discount factor of Reinforcement Learning

        :param learning_rate: learning rate of optimizer
        :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.gamma = None
        self.states = None
        self.device = None
        self.traj_list = None
        self.action_dim = None
        self.reward_scale = None
        self.if_off_policy = True

        self.env_num = env_num
        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.clip_grad_norm = 4.0
        # self.amp_scale = None  # automatic mixed precision

        """attribute"""
        self.explore_env = None
        self.get_obj_critic = None

        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = (
            self.cri_target
        ) = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = (
            self.act_target
        ) = self.if_use_act_target = self.act_optim = self.ClassAct = None

        assert isinstance(gpu_id, int)
        assert isinstance(env_num, int)
        assert isinstance(net_dim, int)
        assert isinstance(state_dim, int)
        assert isinstance(action_dim, int)
        assert isinstance(if_per_or_gae, bool)
        assert isinstance(gamma, float)
        assert isinstance(reward_scale, float)
        assert isinstance(learning_rate, float)

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param reward_scale: scale the reward to get a appropriate scale Q value
        :param gamma: the discount factor of Reinforcement Learning

        :param learning_rate: learning rate of optimizer
        :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.traj_list = [list() for _ in range(env_num)]
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu"
        )

        self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(
            self.device
        )
        self.act = (
            self.ClassAct(net_dim, state_dim, action_dim).to(self.device)
            if self.ClassAct
            else self.cri
        )
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = (
            torch.optim.Adam(self.act.parameters(), learning_rate)
            if self.ClassAct
            else self.cri
        )
        # del self.ClassCri, self.ClassAct

        assert isinstance(if_per_or_gae, bool)
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Select continuous actions for exploration

        :param state: states.shape==(batch_size, state_dim, )
        :return: actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """

        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(
                -1, 1
            )
        return action.detach().cpu()

    def explore_one_env(self, env, target_step: int) -> list:
        """actor explores in single Env, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step()
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        state = self.states[0]
        traj = list()
        for _ in range(target_step):
            ten_state = torch.as_tensor(state, dtype=torch.float32)
            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()
            next_s, reward, done, _ = env.step(action)

            ten_other = torch.empty(2 + self.action_dim)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2:] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s

        self.states[0] = state

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [
            (traj_state, traj_other),
        ]
        return self.convert_trajectory(traj_list)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step: int) -> list:
        """actor explores in VectorEnv, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step(). It should be a vector env.
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        ten_states = self.states

        traj = list()
        for _ in range(target_step):
            ten_actions = self.select_actions(ten_states)
            ten_next_states, ten_rewards, ten_dones = env.step(ten_actions)

            ten_others = torch.cat(
                (ten_rewards.unsqueeze(0), ten_dones.unsqueeze(0), ten_actions)
            )
            traj.append((ten_states, ten_others))
            ten_states = ten_next_states

        self.states = ten_states

        # traj = [(env_ten, ...), ...], env_ten = (env1_ten, env2_ten, ...)
        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [
            (traj_state[:, env_i, :], traj_other[:, env_i, :])
            for env_i in range(len(self.states))
        ]
        # traj_list = [traj_env_0, ...], traj_env_0 = (ten_state, ten_other)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]

    def update_net(
        self,
        buffer,
        batch_size: int,
        repeat_times: float,
        soft_update_tau: float,
    ) -> tuple:
        """update the neural network by sampling batch data from ReplayBuffer

        :param buffer: Experience replay buffer
        :param batch_size: sample batch_size of data for Stochastic Gradient Descent
        :param repeat_times: `batch_sampling_times = int(target_step * repeat_times / batch_size)`
        :param soft_update_tau: soft target update: `target_net = target_net * (1-tau) + current_net * tau`,
        """

    def optim_update(
        self, optimizer, objective, params
    ):  # plan todo params generator -> list
        """minimize the optimization objective via update the network parameters

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        :param params: `params = net.parameters()` the network parameters which need to be updated.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(params, max_norm=self.clip_grad_norm)
        optimizer.step()

    # def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
    #     """minimize the optimization objective via update the network parameters
    #
    #     amp: Automatic Mixed Precision
    #
    #     :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
    #     :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
    #     :param params: `params = net.parameters()` the network parameters which need to be updated.
    #     """
    #     # self.amp_scale = torch.cuda.amp.GradScaler()
    #
    #     optimizer.zero_grad()
    #     self.amp_scale.scale(objective).backward()  # loss.backward()
    #     self.amp_scale.unscale_(optimizer)  # amp
    #
    #     # from torch.nn.utils import clip_grad_norm_
    #     # clip_grad_norm_(model.parameters(), max_norm=3.0)  # amp, clip_grad_norm_
    #     self.amp_scale.step(optimizer)  # optimizer.step()
    #     self.amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update target network via current network

        :param target_net: update target network via current network to make training more stable.
        :param current_net: current network update via an optimizer
        :param tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [
            ("actor", self.act),
            ("act_target", self.act_target),
            ("act_optim", self.act_optim),
            ("critic", self.cri),
            ("cri_target", self.cri_target),
            ("cri_optim", self.cri_optim),
        ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def convert_trajectory(self, traj_list: list) -> list:  # off-policy
        """convert trajectory (env exploration type) to trajectory (replay buffer type)

        convert `other = concat((      reward, done, ...))`
        to      `other = concat((scale_reward, mask, ...))`

        :param traj_list: `traj_list = [(tensor_state, other_state), ...]`
        :return: `traj_list = [(tensor_state, other_state), ...]`
        """
        for ten_state, ten_other in traj_list:
            ten_other[:, 0] = ten_other[:, 0] * self.reward_scale  # ten_reward
            ten_other[:, 1] = (
                1.0 - ten_other[:, 1]
            ) * self.gamma  # ten_mask = (1.0 - ary_done) * gamma
        return traj_list


"""Value-based Methods (Q network)"""


class AgentDQN(AgentBase):  # [ElegantRL.2021.10.25]
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = (
            None  # self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        )
        self.if_use_dueling = (
            True  # self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        )
        self.explore_rate = (
            0.25  # the probability of choosing action randomly in epsilon-greedy
        )

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        self.ClassCri = QNetDuel if self.if_use_dueling else QNet
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(
        self, states: torch.Tensor
    ) -> torch.Tensor:  # for discrete action space
        """Select discrete actions for exploration

        `tensor states` states.shape==(batch_size, state_dim, )
        return `tensor a_ints` a_ints.shape==(batch_size, )
        """
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_ints = torch.randint(
                self.action_dim, size=states.shape[0]
            )  # choosing action randomly
        else:
            actions = self.act(states.to(self.device))
            a_ints = actions.argmax(dim=1)
        return a_ints.detach().cpu()

    def explore_one_env(self, env, target_step) -> list:
        traj = list()
        state = self.states[0]
        for _ in range(target_step):
            ten_state = torch.as_tensor(state, dtype=torch.float32)
            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()  # isinstance(action, int)
            next_s, reward, done, _ = env.step(action)

            ten_other = torch.empty(2 + 1)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s
        self.states[0] = state

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [
            (traj_state, traj_other),
        ]
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]

    def explore_vec_env(self, env, target_step) -> list:
        ten_states = self.states

        traj = list()
        for _ in range(target_step):
            ten_actions = self.select_actions(ten_states)
            ten_next_states, ten_rewards, ten_dones = env.step(ten_actions)

            ten_others = torch.cat(
                (
                    ten_rewards.unsqueeze(0),
                    ten_dones.unsqueeze(0),
                    ten_actions.unsqueeze(0),
                )
            )
            traj.append((ten_states, ten_others))
            ten_states = ten_next_states

        self.states = ten_states

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [
            (traj_state[:, env_i, :], traj_other[:, env_i, :])
            for env_i in range(len(self.states))
        ]
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = q_value = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, q_value = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
        return obj_critic.item(), q_value.mean().item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, q_value

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            (
                reward,
                mask,
                action,
                state,
                next_s,
                is_weights,
            ) = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s).max(dim=1, keepdim=True)[0]
            q_label = reward + mask * next_q

        q_value = self.cri(state).gather(1, action.long())
        td_error = self.criterion(
            q_value, q_label
        )  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


class AgentDoubleDQN(AgentDQN):  # [ElegantRL.2021.10.25]
    def __init__(self):
        AgentDQN.__init__(self)
        self.soft_max = torch.nn.Softmax(dim=1)

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        self.ClassCri = QNetTwinDuel if self.if_use_dueling else QNetTwin
        AgentDQN.init(
            self,
            net_dim,
            state_dim,
            action_dim,
            learning_rate,
            reward_scale,
            gamma,
            if_per_or_gae,
            env_num,
            gpu_id,
        )

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(
        self, states: torch.Tensor
    ) -> torch.Tensor:  # for discrete action space
        actions = self.act(states.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            a_prob = self.soft_max(actions)
            a_ints = torch.multinomial(a_prob, num_samples=1, replacement=True)[:, 0]
            # a_int = rd.choice(self.action_dim, prob=a_prob)  # numpy version
        else:
            a_ints = actions.argmax(dim=1)
        return a_ints.detach().cpu()

    def get_obj_critic_raw(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(
                dim=1, keepdim=True
            )[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, q1

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            (
                reward,
                mask,
                action,
                state,
                next_s,
                is_weights,
            ) = buffer.sample_batch(batch_size)
            next_q = torch.min(*self.cri_target.get_q1_q2(next_s)).max(
                dim=1, keepdim=True
            )[0]
            q_label = reward + mask * next_q

        q1, q2 = [qs.gather(1, action.long()) for qs in self.act.get_q1_q2(state)]
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q1


"""Actor-Critic Methods (Policy Gradient)"""


class AgentDDPG(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = Actor
        self.ClassCri = Critic
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.3  # explore noise of action (OrnsteinUhlenbeckNoise)
        self.ou_noise = None

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )
        self.ou_noise = OrnsteinUhlenbeckNoise(
            size=action_dim, sigma=self.explore_noise
        )

        if if_per_or_gae:
            self.criterion = torch.nn.SmoothL1Loss(
                reduction="none" if if_per_or_gae else "mean"
            )
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(
                reduction="none" if if_per_or_gae else "mean"
            )
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            ou_noise = torch.as_tensor(
                self.ou_noise(), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = (action + ou_noise).clamp(-1, 1)
        return action.detach().cpu()

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> (float, float):
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for _ in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri(state, action_pg).mean()
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q
        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            (
                reward,
                mask,
                action,
                state,
                next_s,
                is_weights,
            ) = buffer.sample_batch(batch_size)
            next_q = self.cri_target(next_s, self.act_target(next_s))
            q_label = reward + mask * next_q

        q_value = self.cri(state, action)
        td_error = self.criterion(
            q_value, q_label
        )  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentTD3(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = Actor
        self.ClassCri = CriticTwin
        self.if_use_cri_target = True
        self.if_use_act_target = True

        self.explore_noise = 0.1  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())

            if update_c % self.update_freq == 0:  # delay update
                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.cri_target(
                    state, action_pg
                ).mean()  # use cri_target is more stable than cri
                self.optim_update(self.act_optim, obj_actor, self.act.parameters())
                if self.if_use_cri_target:
                    self.soft_update(self.cri_target, self.cri, soft_update_tau)
                if self.if_use_act_target:
                    self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(
                next_s, self.policy_noise
            )  # policy noise
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics
            q_label = reward + mask * next_q
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(
            q2, q_label
        )  # twin critics
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        """Prioritized Experience Replay

        Contributor: Github GyChou
        """
        with torch.no_grad():
            (
                reward,
                mask,
                action,
                state,
                next_s,
                is_weights,
            ) = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(
                next_s, self.policy_noise
            )  # policy noise
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentSAC(AgentBase):  # [ElegantRL.2021.10.25]
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = CriticTwin
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False

        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )

        self.alpha_log = torch.tensor(
            (-np.log(action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )  # trainable parameter
        self.alpha_optim = torch.optim.Adam((self.alpha_log,), lr=learning_rate)
        self.target_entropy = np.log(action_dim)

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        state = state.to(self.device)
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            actions = self.act.get_action(state)
        else:
            actions = self.act(state)
        return actions.detach().cpu()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        obj_actor = None
        alpha = None
        for _ in range(int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.0025 * obj_critic.item()
            )  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            """objective of alpha (temperature parameter automatic adjustment)"""
            action_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (
                self.alpha_log * (logprob - self.target_entropy).detach()
            ).mean()
            self.optim_update(self.alpha_optim, obj_alpha, self.alpha_log)

            """objective of actor"""
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-20, 2).detach()
            obj_actor = -(
                torch.min(*self.cri.get_q1_q2(state, action_pg)) + logprob * alpha
            ).mean()
            # use self.cri_target.get_q1_q2 in above code for more stable training.
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())

            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return self.obj_critic, obj_actor.item(), alpha.item()

    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)
        q1, q2 = self.cri.get_q1_q2(state, action)
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size, alpha):
        with torch.no_grad():
            (
                reward,
                mask,
                action,
                state,
                next_s,
                is_weights,
            ) = buffer.sample_batch(batch_size)

            next_a, next_log_prob = self.act_target.get_action_logprob(
                next_s
            )  # stochastic policy
            next_q = torch.min(
                *self.cri_target.get_q1_q2(next_s, next_a)
            )  # twin critics

            q_label = reward + mask * (next_q + next_log_prob * alpha)

        q1, q2 = self.cri.get_q1_q2(state, action)
        td_error = self.criterion(q1, q_label) + self.criterion(q2, q_label)
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, state


class AgentModSAC(
    AgentSAC
):  # Modified SAC using reliable_lambda and TTUR (Two Time-scale Update Rule)
    def __init__(self):
        AgentSAC.__init__(self)
        self.if_use_act_target = True
        self.if_use_cri_target = True
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len * repeat_times / batch_size)):
            alpha = self.alpha_log.exp()

            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.0025 * obj_critic.item()
            )  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            """objective of alpha (temperature parameter automatic adjustment)"""
            obj_alpha = (
                self.alpha_log * (logprob - self.target_entropy).detach()
            ).mean()
            self.optim_update(self.alpha_optim, obj_alpha, self.alpha_log)
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(*self.cri.get_q1_q2(state, a_noise_pg))
                obj_actor = -(q_value_pg + logprob * alpha).mean()
                self.optim_update(self.act_optim, obj_actor, self.act.parameters())
                if self.if_use_act_target:
                    self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()


class AgentPPO(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ActorPPO
        self.ClassCri = CriticPPO

        self.if_off_policy = False
        self.ratio_clip = 0.2  # could be 0.00 ~ 0.50 ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02  # could be 0.00~0.10
        self.lambda_a_value = 1.00  # could be 0.25~8.00, the lambda of advantage value
        self.lambda_gae_adv = 0.98  # could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.get_reward_sum = (
            None  # self.get_reward_sum_gae if if_use_gae else self.get_reward_sum_raw
        )

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )
        self.traj_list = [list() for _ in range(env_num)]
        self.env_num = env_num

        if if_per_or_gae:  # if_use_gae
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def select_actions(self, state: torch.Tensor) -> tuple:
        """
        `tensor state` state.shape = (batch_size, state_dim)
        return `tensor action` action.shape = (batch_size, action_dim)
        return `tensor noise` noise.shape = (batch_size, action_dim)
        """
        state = state.to(self.device)
        action, noise = self.act.get_action(state)
        return action.detach().cpu(), noise.detach().cpu()

    def explore_one_env(self, env, target_step):
        state = self.states[0]

        last_done = 0
        traj = list()
        for step_i in range(target_step):
            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_actions, ten_noises = self.select_actions(ten_states)
            action = ten_actions[0].numpy()
            next_s, reward, done, _ = env.step(np.tanh(action))

            traj.append((ten_states, reward, done, ten_actions, ten_noises))
            if done:
                state = env.reset()
                last_done = step_i
            else:
                state = next_s

        self.states[0] = state

        traj_list = self.splice_trajectory(
            [
                traj,
            ],
            [
                last_done,
            ],
        )
        return self.convert_trajectory(traj_list)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step):
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in range(target_step):
            ten_actions, ten_noises = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_actions.tanh())

            for env_i in range(env_num):
                traj_list[env_i].append(
                    (
                        ten_states[env_i],
                        ten_rewards[env_i],
                        ten_dones[env_i],
                        ten_actions[env_i],
                        ten_noises[env_i],
                    )
                )
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [
                ten.to(self.device) for ten in buffer
            ]

            """get buf_r_sum, buf_logprob"""
            bs = 2**10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i : i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                self.lambda_a_value / (buf_adv_v.std() + 1e-5)
            )
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        update_times = int(buf_len / batch_size * repeat_times)
        for update_i in range(1, update_times + 1):
            indices = torch.randint(
                buf_len,
                size=(batch_size,),
                requires_grad=False,
                device=self.device,
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return (
            obj_critic.item(),
            obj_actor.item(),
            a_std_log.item(),
        )  # logging_tuple

    def get_reward_sum_raw(
        self, buf_len, buf_reward, buf_mask, buf_value
    ) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # reward sum

        pre_r_sum = 0
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
        buf_adv_v = buf_r_sum - buf_value[:, 0]
        return buf_r_sum, buf_adv_v

    def get_reward_sum_gae(
        self, buf_len, ten_reward, ten_mask, ten_value
    ) -> (torch.Tensor, torch.Tensor):
        buf_r_sum = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # old policy value
        buf_adv_v = torch.empty(
            buf_len, dtype=torch.float32, device=self.device
        )  # advantage value

        pre_r_sum = 0
        pre_adv_v = 0  # advantage value of previous step
        ten_bool = torch.not_equal(ten_mask, 0).float()
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = ten_reward[i] + ten_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_adv_v[i] = ten_reward[i] + ten_bool[i] * (
                pre_adv_v - ten_value[i]
            )  # todo need to check
            pre_adv_v = ten_value[i] + buf_adv_v[i] * self.lambda_gae_adv
        return buf_r_sum, buf_adv_v

    def splice_trajectory(self, traj_list, last_done_list):
        for env_i in range(self.env_num):
            last_done = last_done_list[env_i]
            traj_temp = traj_list[env_i]

            traj_list[env_i] = self.traj_list[env_i] + traj_temp[: last_done + 1]
            self.traj_list[env_i] = traj_temp[last_done:]
        return traj_list

    def convert_trajectory(self, traj_list):
        for traj in traj_list:
            temp = list(map(list, zip(*traj)))  # 2D-list transpose

            ten_state = torch.stack(temp[0])
            ten_reward = (
                torch.as_tensor(temp[1], dtype=torch.float32) * self.reward_scale
            )
            ten_mask = (
                1.0 - torch.as_tensor(temp[2], dtype=torch.float32)
            ) * self.gamma
            ten_action = torch.stack(temp[3])
            ten_noise = torch.stack(temp[4])

            traj[:] = (ten_state, ten_reward, ten_mask, ten_action, ten_noise)
        return traj_list


class AgentDiscretePPO(AgentPPO):
    def __init__(self):
        AgentPPO.__init__(self)
        self.ClassAct = ActorDiscretePPO

    def explore_one_env(self, env, target_step):
        state = self.states[0]

        last_done = 0
        traj = list()
        for step_i in range(target_step):
            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            a_int = ten_a_ints[0].numpy()
            next_s, reward, done, _ = env.step(a_int)  # only different

            traj.append((ten_states, reward, done, ten_a_ints, ten_probs))
            if done:
                state = env.reset()
                last_done = step_i
            else:
                state = next_s

        self.states[0] = state

        traj_list = self.splice_trajectory(
            [
                traj,
            ],
            [
                last_done,
            ],
        )
        return self.convert_trajectory(traj_list)

    def explore_vec_env(self, env, target_step):
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in range(target_step):
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_a_ints.numpy())

            for env_i in range(env_num):
                traj_list[env_i].append(
                    (
                        ten_states[env_i],
                        ten_rewards[env_i],
                        ten_dones[env_i],
                        ten_a_ints[env_i],
                        ten_probs[env_i],
                    )
                )
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]


class AgentA2C(AgentPPO):  # A2C.2015, PPO.2016
    def __init__(self):
        AgentPPO.__init__(self)
        print(
            "| AgentA2C: A2C or A3C is worse than PPO. We provide AgentA2C code just for teaching."
            "| Without TrustRegion, A2C needs special hyper-parameters, such as smaller repeat_times."
        )

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_reward, buf_mask, buf_action, buf_noise = [
                ten.to(self.device) for ten in buffer
            ]

            """get buf_r_sum, buf_logprob"""
            bs = 2**10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i : i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            # buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                self.lambda_a_value / (buf_adv_v.std() + 1e-5)
            )
            # buf_adv_v: advantage_value in ReplayBuffer
            del buf_noise, buffer[:]

        obj_critic = None
        obj_actor = None
        update_times = int(buf_len / batch_size * repeat_times)
        for update_i in range(1, update_times + 1):
            indices = torch.randint(
                buf_len,
                size=(batch_size,),
                requires_grad=False,
                device=self.device,
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]
            action = buf_action[indices]
            # logprob = buf_logprob[indices]

            """A2C: Advantage function"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            obj_actor = (
                -(adv_v * new_logprob.exp()).mean() + obj_entropy * self.lambda_entropy
            )
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return (
            obj_critic.item(),
            obj_actor.item(),
            a_std_log.item(),
        )  # logging_tuple


class AgentDiscreteA2C(AgentA2C):
    def __init__(self):
        AgentA2C.__init__(self)
        self.ClassAct = ActorDiscretePPO

    def explore_one_env(self, env, target_step):
        state = self.states[0]

        last_done = 0
        traj = list()
        for step_i in range(target_step):
            ten_states = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            a_int = ten_a_ints[0].numpy()
            next_s, reward, done, _ = env.step(a_int)  # only different

            traj.append((ten_states, reward, done, ten_a_ints, ten_probs))
            if done:
                state = env.reset()
                last_done = step_i
            else:
                state = next_s

        self.states[0] = state

        traj_list = self.splice_trajectory(
            [
                traj,
            ],
            [
                last_done,
            ],
        )
        return self.convert_trajectory(traj_list)

    def explore_vec_env(self, env, target_step):
        ten_states = self.states

        env_num = len(self.traj_list)
        traj_list = [list() for _ in range(env_num)]  # [traj_env_0, ..., traj_env_i]
        last_done_list = [0 for _ in range(env_num)]

        for step_i in range(target_step):
            ten_a_ints, ten_probs = self.select_actions(ten_states)
            tem_next_states, ten_rewards, ten_dones = env.step(ten_a_ints.numpy())

            for env_i in range(env_num):
                traj_list[env_i].append(
                    (
                        ten_states[env_i],
                        ten_rewards[env_i],
                        ten_dones[env_i],
                        ten_a_ints[env_i],
                        ten_probs[env_i],
                    )
                )
                if ten_dones[env_i]:
                    last_done_list[env_i] = step_i

            ten_states = tem_next_states

        self.states = ten_states

        traj_list = self.splice_trajectory(traj_list, last_done_list)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]


class AgentStep1AC(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ActorBiConv
        self.ClassCri = CriticBiConv
        self.if_use_cri_target = False
        self.if_use_act_target = False
        self.explore_noise = 2**-8
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )
        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.MSELoss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw
        self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act.get_action(state.to(self.device), self.explore_noise)
        return action.detach().cpu()

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> (float, float):
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            """objective of critic (loss function of critic)"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.obj_critic = (
                0.99 * self.obj_critic + 0.01 * obj_critic.item()
            )  # for reliable_lambda
            self.optim_update(self.cri_optim, obj_critic, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

            obj_actor = -self.cri(state, self.act(state)).mean()  # policy gradient
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            q_label, action, state = buffer.sample_batch_one_step(batch_size)

        q_value = self.cri(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            q_label, action, state, is_weights = buffer.sample_batch_one_step(
                batch_size
            )

        q_value = self.cri(state, action)
        td_error = self.criterion(
            q_value, q_label
        )  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


"""Actor-Critic Methods (Parameter Sharing)"""


class AgentShareAC(AgentBase):  # IAC (InterAC) waiting for check
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = ShareDPG  # self.Act = None

        self.explore_noise = 0.2  # standard deviation of explore noise
        self.policy_noise = 0.4  # standard deviation of policy noise
        self.update_freq = 2**7  # delay update frequency, for hard target update
        self.avg_loss_c = (-np.log(0.5)) ** 0.5  # old version reliable_lambda

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        reliable_lambda = None
        k = 1.0 + buffer.now_len / buffer.max_len
        batch_size_ = int(batch_size * k)
        for i in range(int(buffer.now_len / batch_size * repeat_times)):
            with torch.no_grad():
                reward, mask, action, state, next_state = buffer.sample_batch(
                    batch_size_
                )

                next_q_label, next_action = self.cri_target.next_q_action(
                    state, next_state, self.policy_noise
                )
                q_label = reward + mask * next_q_label

            """obj_critic"""
            q_eval = self.cri.critic(state, action)
            obj_critic = self.criterion(q_eval, q_label)

            """auto reliable lambda"""
            self.avg_loss_c = (
                0.995 * self.avg_loss_c + 0.005 * obj_critic.item() / 2
            )  # soft update, twin critics
            reliable_lambda = np.exp(-self.avg_loss_c**2)

            """actor correction term"""
            actor_term = self.criterion(self.cri(next_state), next_action)

            if i % repeat_times == 0:
                """actor obj"""
                action_pg = self.cri(state)  # policy gradient
                obj_actor = -self.cri_target.critic(
                    state, action_pg
                ).mean()  # policy gradient
                # NOTICE! It is very important to use act_target.critic here instead act.critic
                # Or you can use act.critic.deepcopy(). Whatever you cannot use act.critic directly.

                obj_united = (
                    obj_critic
                    + actor_term * (1 - reliable_lambda)
                    + obj_actor * (reliable_lambda * 0.5)
                )
            else:
                obj_united = obj_critic + actor_term * (1 - reliable_lambda)

            """united loss"""
            self.optim_update(self.cri_optim, obj_united, self.cri.parameters())
            if i % self.update_freq == self.update_freq and reliable_lambda > 0.1:
                self.cri_target.load_state_dict(
                    self.cri.state_dict()
                )  # Hard Target Update

        return obj_critic.item(), obj_actor.item(), reliable_lambda


class AgentShareSAC(AgentSAC):  # Integrated Soft Actor-Critic
    def __init__(self):
        AgentSAC.__init__(self)
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
        self.cri_optim = None

        self.target_entropy = None
        self.alpha_log = None

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.alpha_log = torch.tensor(
            (-np.log(action_dim) * np.e,),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )  # trainable parameter
        self.target_entropy = np.log(action_dim)
        self.act = self.cri = ShareSPG(net_dim, state_dim, action_dim).to(self.device)
        self.act_target = self.cri_target = deepcopy(self.act)

        self.cri_optim = torch.optim.Adam(
            [
                {
                    "params": self.act.enc_s.parameters(),
                    "lr": learning_rate * 1.5,
                },
                {
                    "params": self.act.enc_a.parameters(),
                },
                {
                    "params": self.act.net.parameters(),
                    "lr": learning_rate * 1.5,
                },
                {
                    "params": self.act.dec_a.parameters(),
                },
                {
                    "params": self.act.dec_d.parameters(),
                },
                {
                    "params": self.act.dec_q1.parameters(),
                },
                {
                    "params": self.act.dec_q2.parameters(),
                },
                {"params": (self.alpha_log,)},
            ],
            lr=learning_rate,
        )

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.SmoothL1Loss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> tuple:  # 1111
        buffer.update_now_len()

        obj_actor = None
        update_a = 0
        alpha = None
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            alpha = self.alpha_log.exp()

            """objective of critic"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size, alpha)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.0025 * obj_critic.item()
            )  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda

            """objective of alpha (temperature parameter automatic adjustment)"""
            a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
            obj_alpha = (
                self.alpha_log
                * (logprob - self.target_entropy).detach()
                * reliable_lambda
            ).mean()
            with torch.no_grad():
                self.alpha_log[:] = self.alpha_log.clamp(-16, 2).detach()

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                q_value_pg = torch.min(
                    *self.act_target.get_q1_q2(state, a_noise_pg)
                ).mean()  # twin critics
                obj_actor = -(
                    q_value_pg + logprob * alpha.detach()
                ).mean()  # policy gradient

                obj_united = obj_critic + obj_alpha + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic + obj_alpha

            self.optim_update(self.cri_optim, obj_united, self.cri.parameters())
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return self.obj_critic, obj_actor.item(), alpha.item()


class AgentSharePPO(AgentPPO):
    def __init__(self):
        AgentPPO.__init__(self)
        self.obj_c = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        if if_per_or_gae:
            self.get_reward_sum = self.get_reward_sum_gae
        else:
            self.get_reward_sum = self.get_reward_sum_raw

        self.act = self.cri = SharePPO(state_dim, action_dim, net_dim).to(self.device)

        self.cri_optim = torch.optim.Adam(
            [
                {
                    "params": self.act.enc_s.parameters(),
                    "lr": learning_rate * 0.9,
                },
                {
                    "params": self.act.dec_a.parameters(),
                },
                {
                    "params": self.act.a_std_log,
                },
                {
                    "params": self.act.dec_q1.parameters(),
                },
                {
                    "params": self.act.dec_q2.parameters(),
                },
            ],
            lr=learning_rate,
        )
        self.criterion = torch.nn.SmoothL1Loss()

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [
                ten.to(self.device) for ten in buffer
            ]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

            """get buf_r_sum, buf_logprob"""
            bs = 2**10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i : i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                self.lambda_a_value / torch.std(buf_adv_v) + 1e-5
            )
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(
                buf_len,
                size=(batch_size,),
                requires_grad=False,
                device=self.device,
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]  # advantage value
            action = buf_action[indices]
            logprob = buf_logprob[indices]

            """PPO: Surrogate objective of Trust Region"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = adv_v * ratio
            surrogate2 = adv_v * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            obj_united = obj_critic + obj_actor
            self.optim_update(self.cri_optim, obj_united, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return (
            obj_critic.item(),
            obj_actor.item(),
            a_std_log.item(),
        )  # logging_tuple


class AgentShareA2C(AgentSharePPO):
    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        with torch.no_grad():
            buf_len = buffer[0].shape[0]
            buf_state, buf_action, buf_noise, buf_reward, buf_mask = [
                ten.to(self.device) for ten in buffer
            ]
            # (ten_state, ten_action, ten_noise, ten_reward, ten_mask) = buffer

            """get buf_r_sum, buf_logprob"""
            bs = 2**10  # set a smaller 'BatchSize' when out of GPU memory.
            buf_value = [
                self.cri_target(buf_state[i : i + bs]) for i in range(0, buf_len, bs)
            ]
            buf_value = torch.cat(buf_value, dim=0)
            # buf_logprob = self.act.get_old_logprob(buf_action, buf_noise)

            buf_r_sum, buf_adv_v = self.get_reward_sum(
                buf_len, buf_reward, buf_mask, buf_value
            )  # detach()
            buf_adv_v = (buf_adv_v - buf_adv_v.mean()) * (
                self.lambda_a_value / torch.std(buf_adv_v) + 1e-5
            )
            # buf_adv_v: buffer data of adv_v value
            del buf_noise, buffer[:]

        obj_critic = obj_actor = None
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(
                buf_len,
                size=(batch_size,),
                requires_grad=False,
                device=self.device,
            )

            state = buf_state[indices]
            r_sum = buf_r_sum[indices]
            adv_v = buf_adv_v[indices]  # advantage value
            action = buf_action[indices]
            # logprob = buf_logprob[indices]

            """A2C: Advantage function"""
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state, action
            )  # it is obj_actor
            obj_actor = (
                -(adv_v * new_logprob.exp()).mean() + obj_entropy * self.lambda_entropy
            )
            self.optim_update(self.act_optim, obj_actor, self.act.parameters())

            value = self.cri(state).squeeze(
                1
            )  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)

            obj_united = obj_critic + obj_actor
            self.optim_update(self.cri_optim, obj_united, self.cri.parameters())
            if self.if_use_cri_target:
                self.soft_update(self.cri_target, self.cri, soft_update_tau)

        a_std_log = getattr(self.act, "a_std_log", torch.zeros(1)).mean()
        return (
            obj_critic.item(),
            obj_actor.item(),
            a_std_log.item(),
        )  # logging_tuple


class AgentShareStep1AC(AgentBase):
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassAct = ShareBiConv
        self.ClassCri = self.ClassAct
        self.if_use_cri_target = True
        self.if_use_act_target = True
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda

    def init(
        self,
        net_dim=256,
        state_dim=8,
        action_dim=2,
        reward_scale=1.0,
        gamma=0.99,
        learning_rate=1e-4,
        if_per_or_gae=False,
        env_num=1,
        gpu_id=0,
    ):
        AgentBase.init(
            self,
            net_dim=net_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            reward_scale=reward_scale,
            gamma=gamma,
            learning_rate=learning_rate,
            if_per_or_gae=if_per_or_gae,
            env_num=env_num,
            gpu_id=gpu_id,
        )
        self.act = self.cri = self.ClassAct(net_dim, state_dim, action_dim).to(
            self.device
        )
        if self.if_use_act_target:
            self.act_target = self.cri_target = deepcopy(self.act)
        else:
            self.act_target = self.cri_target = self.act

        self.cri_optim = torch.optim.Adam(
            [
                {
                    "params": self.act.enc_s.parameters(),
                    "lr": learning_rate * 1.5,
                },
                {
                    "params": self.act.enc_a.parameters(),
                },
                {
                    "params": self.act.mid_n.parameters(),
                    "lr": learning_rate * 1.5,
                },
                {
                    "params": self.act.dec_a.parameters(),
                },
                {
                    "params": self.act.dec_q.parameters(),
                },
            ],
            lr=learning_rate,
        )
        self.act_optim = self.cri_optim

        if if_per_or_gae:  # if_use_per
            self.criterion = torch.nn.MSELoss(reduction="none")
            self.get_obj_critic = self.get_obj_critic_per
        else:
            self.criterion = torch.nn.MSELoss(reduction="mean")
            self.get_obj_critic = self.get_obj_critic_raw

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        action = self.act.get_action(state.to(self.device), self.explore_noise)
        return action.detach().cpu()

    def update_net(
        self, buffer, batch_size, repeat_times, soft_update_tau
    ) -> (float, float):
        buffer.update_now_len()

        obj_critic = None
        obj_actor = None
        update_a = 0
        for update_c in range(1, int(buffer.now_len / batch_size * repeat_times)):
            """objective of critic"""
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.obj_critic = (
                0.995 * self.obj_critic + 0.005 * obj_critic.item()
            )  # for reliable_lambda
            reliable_lambda = np.exp(-self.obj_critic**2)  # for reliable_lambda

            """objective of actor using reliable_lambda and TTUR (Two Time-scales Update Rule)"""
            if_update_a = update_a / update_c < 1 / (2 - reliable_lambda)
            if if_update_a:  # auto TTUR
                update_a += 1

                action_pg = self.act(state)  # policy gradient
                obj_actor = -self.act_target.critic(state, action_pg).mean()

                obj_united = obj_critic + obj_actor * reliable_lambda
            else:
                obj_united = obj_critic

            self.optim_update(self.act_optim, obj_united, self.act.parameters())
            if self.if_use_act_target:
                self.soft_update(self.act_target, self.act, soft_update_tau)

        return obj_critic.item(), obj_actor.item()

    def get_obj_critic_raw(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            q_label, action, state = buffer.sample_batch_one_step(batch_size)

        q_value = self.act.critic(state, action)
        obj_critic = self.criterion(q_value, q_label)
        return obj_critic, state

    def get_obj_critic_per(self, buffer, batch_size):
        with torch.no_grad():
            # reward, mask, action, state, next_s, is_weights = buffer.sample_batch(batch_size)
            q_label, action, state, is_weights = buffer.sample_batch_one_step(
                batch_size
            )

        q_value = self.act.critic(state, action)
        td_error = self.criterion(
            q_value, q_label
        )  # or td_error = (q_value - q_label).abs()
        obj_critic = (td_error * is_weights).mean()

        buffer.td_error_update(td_error.detach())
        return obj_critic, q_value


"""Utils"""


class OrnsteinUhlenbeckNoise:  # NOT suggest to use it
    def __init__(self, size, theta=0.15, sigma=0.3, ou_noise=0.0, dt=1e-2):
        """The noise of Ornstein-Uhlenbeck Process

        Source: https://github.com/slowbull/DDPG/blob/master/src/explorationnoise.py
        It makes Zero-mean Gaussian Noise more stable.
        It helps agent explore better in a inertial system.
        Don't abuse OU Process. OU process has too much hyper-parameters and over fine-tuning make no sense.

        :int size: the size of noise, noise.shape==(-1, action_dim)
        :float theta: related to the not independent of OU-noise
        :float sigma: related to action noise std
        :float ou_noise: initialize OU-noise
        :float dt: derivative
        """
        self.theta = theta
        self.sigma = sigma
        self.ou_noise = ou_noise
        self.dt = dt
        self.size = size

    def __call__(self) -> float:
        """output a OU-noise

        :return array ou_noise: a noise generated by Ornstein-Uhlenbeck Process
        """
        noise = self.sigma * np.sqrt(self.dt) * rd.normal(size=self.size)
        self.ou_noise -= self.theta * self.ou_noise * self.dt + noise
        return self.ou_noise
