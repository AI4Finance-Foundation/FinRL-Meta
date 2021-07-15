import os
import gym
import torch
import numpy as np
# import numpy.random as rd
from copy import deepcopy

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""


class PreprocessEnv(gym.Wrapper):  # environment wrapper
    def __init__(self, env, if_print=True):
        """Preprocess a standard OpenAI gym environment for training.

        `object env` a standard OpenAI gym environment, it has env.reset() and env.step()
        `bool if_print` print the information of environment. Such as env_name, state_dim ...
        `object data_type` convert state (sometimes float64) to data_type (float32).
        """
        self.env = gym.make(env) if isinstance(env, str) else env
        super().__init__(self.env)

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(self.env, if_print)

        state_avg, state_std = get_avg_std__for_state_norm(self.env_name)
        self.neg_state_avg = -state_avg
        self.div_state_std = 1 / (state_std + 1e-4)

        self.reset = self.reset_norm
        self.step = self.step_norm

    def reset_norm(self) -> np.ndarray:
        """ convert the data type of state from float64 to float32
        do normalization on state

        return `array state` state.shape==(state_dim, )
        """
        state = self.env.reset()
        state = (state + self.neg_state_avg) * self.div_state_std
        return state.astype(np.float32)

    def step_norm(self, action: np.ndarray) -> (np.ndarray, float, bool, dict):
        """convert the data type of state from float64 to float32,
        adjust action range to (-action_max, +action_max)
        do normalization on state

        return `array state`  state.shape==(state_dim, )
        return `float reward` reward of one step
        return `bool done` the terminal of an training episode
        return `dict info` the information save in a dict. OpenAI gym standard. Send a `None` is OK
        """
        state, reward, done, info = self.env.step(action * self.action_max)
        state = (state + self.neg_state_avg) * self.div_state_std
        return state.astype(np.float32), reward, done, info


class PreprocessVecEnv:  # environment wrapper
    def __init__(self, env, env_num, if_print=False):
        env = gym.make(env) if isinstance(env, str) else env

        (self.env_name, self.state_dim, self.action_dim, self.action_max, self.max_step,
         self.if_discrete, self.target_return) = get_gym_env_info(env, if_print=if_print)
        self.device = torch.device('cuda')
        self.env_num = env_num

        self.env_list = [gym.make(self.env_name) for _ in range(env_num)]
        for i in range(env_num):
            [self.env_list[i].reset() for _ in range(i)]  # random seed

        state_avg, state_std = get_avg_std__for_state_norm(self.env_name)
        if state_avg is 0 and state_std is 1:
            self.neg_avg = 0
            self.div_std = 1
        else:
            self.neg_avg = -state_avg
            self.div_std = 1 / (state_std + 1e-4)

    def reset_vec(self):
        if not isinstance(self.neg_avg, torch.Tensor):  # convert np.ndarray to torch.Tensor.cuda()
            self.neg_avg = torch.as_tensor((self.neg_avg,), dtype=torch.float32, device=self.device)
            self.neg_avg = self.neg_avg.repeat((self.env_num, 1))

            self.div_std = torch.as_tensor((self.div_std,), dtype=torch.float32, device=self.device)
            self.div_std = self.div_std.repeat((self.env_num, 1))

        states = [env.reset() for env in self.env_list]
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        states = (states + self.neg_avg) * self.div_std
        return states

    def step_vec(self, actions):
        actions = (actions * self.action_max).detach().cpu().numpy()

        states = list()
        rewards = list()
        dones = list()
        for i in range(self.env_num):
            state, reward, done, _ = self.env_list[i].step(actions[i])
            states.append(self.env_list[i].reset() if done else state)
            rewards.append(reward)
            dones.append(done)

        states, rewards, dones = [torch.as_tensor(t, dtype=torch.float32, device=self.device)
                                  for t in (states, rewards, dones)]
        states = (states + self.neg_avg) * self.div_std
        return states, rewards, dones, {}


def deepcopy_or_rebuild_env(env):
    try:
        env_eval = deepcopy(env)
    except Exception as error:
        print('| deepcopy_or_rebuild_env, error:', error)
        env_eval = PreprocessEnv(env.env_name, if_print=False)
    return env_eval


def get_avg_std__for_state_norm(env_name) -> (np.ndarray, np.ndarray):
    """return the state normalization data: neg_avg and div_std

    ReplayBuffer.print_state_norm() will print `neg_avg` and `div_std`
    You can save these array to here. And PreprocessEnv will load them automatically.
    eg. `state = (state + self.neg_state_avg) * self.div_state_std` in `PreprocessEnv.step_norm()`
    neg_avg = -states.mean()
    div_std = 1/(states.std()+1e-5) or 6/(states.max()-states.min())

    `str env_name` the name of environment that helps to find neg_avg and div_std
    return `array avg` neg_avg.shape=(state_dim)
    return `array std` div_std.shape=(state_dim)
    """
    avg = 0
    std = 1
    if env_name == 'LunarLanderContinuous-v2':
        avg = np.array([1.65470898e-02, -1.29684399e-01, 4.26883133e-03, -3.42124557e-02,
                        -7.39076972e-03, -7.67103031e-04, 1.12640885e+00, 1.12409466e+00])
        std = np.array([0.15094465, 0.29366297, 0.23490797, 0.25931464, 0.21603736,
                        0.25886878, 0.277233, 0.27771219])
    elif env_name == "BipedalWalker-v3":
        avg = np.array([1.42211734e-01, -2.74547996e-03, 1.65104509e-01, -1.33418152e-02,
                        -2.43243194e-01, -1.73886203e-02, 4.24114229e-02, -6.57800099e-02,
                        4.53460692e-01, 6.08022244e-01, -8.64884810e-04, -2.08789053e-01,
                        -2.92092949e-02, 5.04791247e-01, 3.33571745e-01, 3.37325723e-01,
                        3.49106580e-01, 3.70363115e-01, 4.04074671e-01, 4.55838055e-01,
                        5.36685407e-01, 6.70771701e-01, 8.80356865e-01, 9.97987386e-01])
        std = np.array([0.84419678, 0.06317835, 0.16532085, 0.09356959, 0.486594,
                        0.55477525, 0.44076614, 0.85030824, 0.29159821, 0.48093035,
                        0.50323634, 0.48110776, 0.69684234, 0.29161077, 0.06962932,
                        0.0705558, 0.07322677, 0.07793258, 0.08624322, 0.09846895,
                        0.11752805, 0.14116005, 0.13839757, 0.07760469])
    elif env_name == 'ReacherBulletEnv-v0':
        avg = np.array([0.03149641, 0.0485873, -0.04949671, -0.06938662, -0.14157104,
                        0.02433294, -0.09097818, 0.4405931, 0.10299437], dtype=np.float32)
        std = np.array([0.12277275, 0.1347579, 0.14567468, 0.14747661, 0.51311225,
                        0.5199606, 0.2710207, 0.48395795, 0.40876198], dtype=np.float32)
    elif env_name == 'AntBulletEnv-v0':
        avg = np.array([-1.4400886e-01, -4.5074993e-01, 8.5741436e-01, 4.4249415e-01,
                        -3.1593361e-01, -3.4174921e-03, -6.1666980e-02, -4.3752361e-03,
                        -8.9226037e-02, 2.5108769e-03, -4.8667483e-02, 7.4835382e-03,
                        3.6160579e-01, 2.6877613e-03, 4.7474738e-02, -5.0628246e-03,
                        -2.5761038e-01, 5.9789192e-04, -2.1119279e-01, -6.6801407e-03,
                        2.5196713e-01, 1.6556121e-03, 1.0365561e-01, 1.0219718e-02,
                        5.8209229e-01, 7.7563477e-01, 4.8815918e-01, 4.2498779e-01],
                       dtype=np.float32)
        std = np.array([0.04128463, 0.19463477, 0.15422264, 0.16463493, 0.16640785,
                        0.08266512, 0.10606721, 0.07636797, 0.7229637, 0.52585346,
                        0.42947173, 0.20228386, 0.44787514, 0.33257666, 0.6440182,
                        0.38659114, 0.6644085, 0.5352245, 0.45194066, 0.20750992,
                        0.4599643, 0.3846344, 0.651452, 0.39733195, 0.49320385,
                        0.41713253, 0.49984455, 0.4943505], dtype=np.float32)
    elif env_name == 'HumanoidBulletEnv-v0':
        avg = np.array([-1.25880212e-01, -8.51390958e-01, 7.07488894e-01, -5.72232604e-01,
                        -8.76260102e-01, -4.07587215e-02, 7.27005303e-04, 1.23370838e+00,
                        -3.68912554e+00, -4.75829793e-03, -7.42472351e-01, -8.94218776e-03,
                        1.29535913e+00, 3.16205365e-03, 9.13809776e-01, -6.42679911e-03,
                        8.90435696e-01, -7.92571157e-03, 6.54826105e-01, 1.82383414e-02,
                        1.20868635e+00, 2.90832808e-03, -9.96598601e-03, -1.87555347e-02,
                        1.66691601e+00, 7.45300390e-03, -5.63859344e-01, 5.48619963e-03,
                        1.33900166e+00, 1.05895223e-02, -8.30249667e-01, 1.57017610e-03,
                        1.92912612e-02, 1.55787319e-02, -1.19833803e+00, -8.22103582e-03,
                        -6.57119334e-01, -2.40323972e-02, -1.05282271e+00, -1.41856335e-02,
                        8.53593826e-01, -1.73063378e-03, 5.46878874e-01, 5.43514848e-01],
                       dtype=np.float32)
        std = np.array([0.08138401, 0.41358876, 0.33958328, 0.17817754, 0.17003846,
                        0.15247536, 0.690917, 0.481272, 0.40543965, 0.6078898,
                        0.46960834, 0.4825346, 0.38099176, 0.5156369, 0.6534775,
                        0.45825616, 0.38340876, 0.89671516, 0.14449312, 0.47643778,
                        0.21150663, 0.56597894, 0.56706554, 0.49014297, 0.30507362,
                        0.6868296, 0.25598812, 0.52973163, 0.14948095, 0.49912784,
                        0.42137524, 0.42925757, 0.39722264, 0.54846555, 0.5816031,
                        1.139402, 0.29807225, 0.27311933, 0.34721208, 0.38530213,
                        0.4897849, 1.0748593, 0.30166605, 0.30824476], dtype=np.float32)
    # elif env_name == 'MinitaurBulletEnv-v0': # need check
    #     avg = np.array([0.90172989, 1.54730119, 1.24560906, 1.97365306, 1.9413892,
    #                     1.03866835, 1.69646277, 1.18655352, -0.45842347, 0.17845232,
    #                     0.38784456, 0.58572877, 0.91414561, -0.45410697, 0.7591031,
    #                     -0.07008998, 3.43842258, 0.61032482, 0.86689961, -0.33910894,
    #                     0.47030415, 4.5623528, -2.39108079, 3.03559422, -0.36328256,
    #                     -0.20753499, -0.47758384, 0.86756409])
    #     std = np.array([0.34192648, 0.51169916, 0.39370621, 0.55568461, 0.46910769,
    #                     0.28387504, 0.51807949, 0.37723445, 13.16686185, 17.51240024,
    #                     14.80264211, 16.60461412, 15.72930229, 11.38926597, 15.40598346,
    #                     13.03124941, 2.47718145, 2.55088804, 2.35964651, 2.51025567,
    #                     2.66379017, 2.37224904, 2.55892521, 2.41716885, 0.07529733,
    #                     0.05903034, 0.1314812, 0.0221248])
    return avg, std


def get_gym_env_info(env, if_print) -> (str, int, int, int, int, bool, float):
    """get information of a standard OpenAI gym env.

    The DRL algorithm AgentXXX need these env information for building networks and training.

    `object env` a standard OpenAI gym environment, it has env.reset() and env.step()
    `bool if_print` print the information of environment. Such as env_name, state_dim ...
    return `env_name` the environment name, such as XxxXxx-v0
    return `state_dim` the dimension of state
    return `action_dim` the dimension of continuous action; Or the number of discrete action
    return `action_max` the max action of continuous action; action_max == 1 when it is discrete action space
    return `max_step` the steps in an episode. (from env.reset to done). It breaks an episode when it reach max_step
    return `if_discrete` Is this env a discrete action space?
    return `target_return` the target episode return, if agent reach this score, then it pass this game (env).
    """
    assert isinstance(env, gym.Env)

    env_name = getattr(env, 'env_name', None)
    env_name = env.unwrapped.spec.id if env_name is None else None

    state_shape = env.observation_space.shape
    state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

    target_return = getattr(env, 'target_return', None)
    target_return_default = getattr(env.spec, 'reward_threshold', None)
    if target_return is None:
        target_return = target_return_default
    if target_return is None:
        target_return = 2 ** 16

    max_step = getattr(env, 'max_step', None)
    max_step_default = getattr(env, '_max_episode_steps', None)
    if max_step is None:
        max_step = max_step_default
    if max_step is None:
        max_step = 2 ** 10

    if_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if if_discrete:  # make sure it is discrete action space
        action_dim = env.action_space.n
        action_max = int(1)
    elif isinstance(env.action_space, gym.spaces.Box):  # make sure it is continuous action space
        action_dim = env.action_space.shape[0]
        action_max = float(env.action_space.high[0])
        assert not any(env.action_space.high + env.action_space.low)
    else:
        raise RuntimeError('| Please set these value manually: if_discrete=bool, action_dim=int, action_max=1.0')

    print(f"\n| env_name:  {env_name}, action space if_discrete: {if_discrete}"
          f"\n| state_dim: {state_dim:4}, action_dim: {action_dim}, action_max: {action_max}"
          f"\n| max_step:  {max_step:4}, target_return: {target_return}") if if_print else None
    return env_name, state_dim, action_dim, action_max, max_step, if_discrete, target_return


"""Custom environment: Fix Env"""


def fix_car_racing_env(env, frame_num=3, action_num=3) -> gym.Wrapper:  # 2020-12-12
    setattr(env, 'old_step', env.step)  # env.old_step = env.step
    setattr(env, 'env_name', 'CarRacing-Fix')
    setattr(env, 'state_dim', (frame_num, 96, 96))
    setattr(env, 'action_dim', 3)
    setattr(env, 'if_discrete', False)
    setattr(env, 'target_return', 700)  # 900 in default

    setattr(env, 'state_stack', None)  # env.state_stack = None
    setattr(env, 'avg_reward', 0)  # env.avg_reward = 0
    """ cancel the print() in environment
    comment 'car_racing.py' line 233-234: print('Track generation ...
    comment 'car_racing.py' line 308-309: print("retry to generate track ...
    """

    def rgb2gray(rgb):
        # # rgb image -> gray [0, 1]
        # gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114]).astype(np.float32)
        # if norm:
        #     # normalize
        #     gray = gray / 128. - 1.
        # return gray

        state = rgb[:, :, 1]  # show green
        state[86:, 24:36] = rgb[86:, 24:36, 2]  # show red
        state[86:, 72:] = rgb[86:, 72:, 0]  # show blue
        state = (state - 128).astype(np.float32) / 128.
        return state

    def decorator_step(env_step):
        def new_env_step(action):
            action = action.copy()
            action[1:] = (action[1:] + 1) / 2  # fix action_space.low

            reward_sum = 0
            done = state = None
            try:
                for _ in range(action_num):
                    state, reward, done, info = env_step(action)
                    state = rgb2gray(state)

                    if done:
                        reward += 100  # don't penalize "die state"
                    if state.mean() > 192:  # 185.0:  # penalize when outside of road
                        reward -= 0.05

                    env.reward_mean = env.reward_mean * 0.95 + reward * 0.05
                    if env.reward_mean <= -0.1:  # done if car don't move
                        done = True

                    reward_sum += reward

                    if done:
                        break
            except Exception as error:
                print(f"| CarRacing-v0 Error 'stack underflow'? {error}")
                reward_sum = 0
                done = True
            env.state_stack.pop(0)
            env.state_stack.append(state)

            return np.array(env.state_stack).flatten(), reward_sum, done, {}

        return new_env_step

    env.step = decorator_step(env.step)

    def decorator_reset(env_reset):
        def new_env_reset():
            state = rgb2gray(env_reset())
            env.state_stack = [state, ] * frame_num
            return np.array(env.state_stack).flatten()

        return new_env_reset

    env.reset = decorator_reset(env.reset)
    return env


def render_car_racing():
    import gym  # gym of OpenAI is not necessary for ElegantRL (even RL)
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'
    env = gym.make('CarRacing-v0')
    env = fix_car_racing_env(env)

    state_dim = env.state_dim

    _state = env.reset()
    import cv2
    action = np.array((0, 1.0, -1.0))
    for i in range(321):
        # action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        # env.render
        show = state.reshape(state_dim)
        show = ((show[0] + 1.0) * 128).astype(np.uint8)
        cv2.imshow('', show)
        cv2.waitKey(1)
        if done:
            break
        # env.render()


"""Utils"""


def demo_get_video_to_watch_gym_render():
    import cv2  # pip3 install opencv-python
    import gym  # pip3 install gym==0.17 pyglet==1.5.0  # env.render() bug in gym==0.18, pyglet==1.6
    import torch

    """parameters"""
    env_name = 'LunarLanderContinuous-v2'
    env = PreprocessEnv(env=gym.make(env_name))

    '''initialize agent'''
    agent = None  # means use random action
    if agent is None:  # use random action
        device = None
    else:
        from elegantrl2.agent import AgentPPO
        agent = AgentPPO()  # means use the policy network which saved in cwd
        cwd = f'./{env_name}_{agent.__class__.__name__}/'  # current working directory path

        net_dim = 2 ** 9  # 2 ** 7
        state_dim = env.state_dim
        action_dim = env.action_dim

        agent.init(net_dim, state_dim, action_dim)
        agent.save_load_model(cwd=cwd, if_save=False)
        device = agent.device

    '''initialize evaluete and env.render()'''
    save_frame_dir = ''  # means don't save video, just open the env.render()
    # save_frame_dir = 'frames'  # means save video in this directory
    if save_frame_dir:
        os.makedirs(save_frame_dir, exist_ok=True)

    state = env.reset()
    episode_return = 0
    step = 0
    for i in range(2 ** 10):
        print(i) if i % 128 == 0 else None
        for j in range(1):
            if agent is None:
                action = env.action_space.sample()
            else:
                s_tensor = torch.as_tensor((state,), dtype=torch.float32, device=device)
                a_tensor = agent.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]  # if use 'with torch.no_grad()', then '.detach()' not need.
            next_state, reward, done, _ = env.step(action)

            episode_return += reward
            step += 1

            if done:
                print(f'{i:>6}, {step:6.0f}, {episode_return:8.3f}, {reward:8.3f}')
                state = env.reset()
                episode_return = 0
                step = 0
            else:
                state = next_state

        if save_frame_dir:
            frame = env.render('rgb_array')
            cv2.imwrite(f'{save_frame_dir}/{i:06}.png', frame)
            cv2.imshow('OpenCV Window', frame)
            cv2.waitKey(1)
        else:
            env.render()
    env.close()

    '''convert frames png/jpg to video mp4/avi using ffmpeg'''
    if save_frame_dir:
        frame_shape = cv2.imread(f'{save_frame_dir}/{3:06}.png').shape
        print(f"frame_shape: {frame_shape}")

        save_video = 'gym_render.mp4'
        os.system(f"| Convert frames to video using ffmpeg. Save in {save_video}")
        os.system(f'ffmpeg -r 60 -f image2 -s {frame_shape[0]}x{frame_shape[1]} '
                  f'-i ./{save_frame_dir}/%06d.png '
                  f'-crf 25 -vb 20M -pix_fmt yuv420p {save_video}')
