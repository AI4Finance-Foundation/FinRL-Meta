import os
import time
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy

from elegantrl2.replay import ReplayBuffer, ReplayBufferMP
from elegantrl2.env import deepcopy_or_rebuild_env

"""[ElegantRL](https://github.com/AI4Finance-LLC/ElegantRL)"""


class Arguments:
    def __init__(self, agent=None, env=None, gpu_id=None, if_on_policy=False):
        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.cwd = None  # current work directory. cwd is None means set it automatically
        self.env = env  # the environment for training
        self.env_eval = None  # the environment for evaluating
        self.gpu_id = gpu_id  # choose the GPU for running. gpu_id is None means set it automatically
        self.worker_num = 2  # the number of rollout workers (larger is not always faster)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training (off-policy)'''
        self.learning_rate = 2 ** -14  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256

        if if_on_policy:  # (on-policy)
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.
        else:
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.max_memo = 2 ** 17  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.

        '''Arguments for evaluate'''
        self.eval_gap = 2 ** 6  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 4
        self.eval_times2 = 2 ** 6
        self.random_seed = 0  # initialize random seed in self.init_before_training()

        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.if_allow_break = True  # allow break training when reach goal (early termination)

    def init_before_training(self, process_id=0):
        if self.agent is None:
            raise RuntimeError('\n| Why agent=None? Assignment args.agent = AgentXXX please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError('\n| Should be agent=AgentXXX() instead of agent=AgentXXX')
        if self.env is None:
            raise RuntimeError('\n| Why env=None? Assignment args.env = XxxEnv() please.')
        if isinstance(self.env, str) or not hasattr(self.env, 'env_name'):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env). It is a Wrapper.')

        '''set None value automatically'''
        if self.gpu_id is None:  # set gpu_id as '0' in default
            self.gpu_id = '0'

        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{self.env.env_name}_{agent_name}_{self.gpu_id}'

        if process_id == 0:
            print(f'| GPU id: {self.gpu_id}, cwd: {self.cwd}')

            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input("PRESS 'y' to REMOVE: {}? ".format(self.cwd)) == 'y')
            if self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print("| Remove history")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        if isinstance(self.gpu_id, tuple) or isinstance(self.gpu_id, list):
            gpu_id_str = str(self.gpu_id)[1:-1]  # for example "0, 1"
        else:
            gpu_id_str = str(self.gpu_id)  # for example "1"
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id_str


'''single process training'''


def train_and_evaluate(args):
    args.init_before_training()

    if True:
        '''basic arguments'''
        cwd = args.cwd
        env = args.env
        agent = args.agent
        gpu_id = args.gpu_id

        '''training arguments'''
        net_dim = args.net_dim
        max_memo = args.max_memo
        break_step = args.break_step
        batch_size = args.batch_size
        target_step = args.target_step
        repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        if_break_early = args.if_allow_break

        gamma = args.gamma
        reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        soft_update_tau = args.soft_update_tau

        '''evaluating arguments'''
        show_gap = args.eval_gap
        eval_times1 = args.eval_times1
        eval_times2 = args.eval_times2
        # if_vec_env = getattr(env, 'env_num', 1) > 1
        env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval
        del args

    '''init: environment'''
    state_dim = env.state_dim
    action_dim = env.action_dim
    env_eval = deepcopy(env) if env_eval is None else deepcopy(env_eval)

    '''init: Agent, Evaluator, '''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae)
    if_on_policy = getattr(agent, 'if_on_policy', False)

    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator

    '''init ReplayBuffer'''
    if if_on_policy:
        buffer = tuple()
    else:
        buffer = ReplayBuffer(max_len=max_memo, state_dim=state_dim, action_dim=action_dim, if_use_per=if_per_or_gae)
        with torch.no_grad():  # update replay buffer
            trajectory_list = explore_before_training(env, target_step, reward_scale, gamma)
            steps = len(trajectory_list)

            state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)
            other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)
            buffer.extend_buffer(state, other)

        agent.state = state
        agent.update_net(buffer, target_step, batch_size, repeat_times)

        # hard update for the first time
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None

        evaluator.total_step += steps

    '''start training'''
    agent.state = env.reset()
    if_train = True
    if if_on_policy:
        while if_train:
            with torch.no_grad():
                if if_on_policy:
                    trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
                    buffer = agent.prepare_buffer(trajectory_list)
                    # buffer = (state, action, r_sum, logprob, advantage)

                    # assert isinstance(buffer, tuple)
                    steps = buffer[2].size(0)  # buffer[2] = r_sum
                    r_exp = buffer[2].mean().item()  # buffer[2] = r_sum
                else:
                    trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
                    state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32)
                    other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32)
                    buffer.extend_buffer(state, other)

                    # assert isinstance(buffer, ReplayBuffer)
                    steps = other.size()[0]
                    r_exp = other[:, 0].mean().item()  # other = (reward, mask, ...)

            logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)

            with torch.no_grad():
                if_reach_goal = evaluator.evaluate_and_save(agent.act, steps, r_exp, logging_tuple)
                if_train = not ((if_break_early and if_reach_goal)
                                or evaluator.total_step > break_step
                                or os.path.exists(f'{cwd}/stop'))

    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')


'''multiple processing training'''


def train_and_evaluate_mp(args):  # multiple processing
    import multiprocessing as mp  # Python built-in multiprocessing library
    process = list()

    pipe_eva = mp.Pipe()
    pipe_exp_list = [mp.Pipe() for _ in range(args.worker_num)]

    process.append(mp.Process(target=mp_learner, args=(args, pipe_eva, pipe_exp_list)))
    process.append(mp.Process(target=mp_evaluator, args=(args, pipe_eva)))
    process.extend([mp.Process(target=mp_worker, args=(args, pipe_exp_list[worker_id], worker_id))
                    for worker_id in range(args.worker_num)])

    [p.start() for p in process]
    process[0].join()
    process_safely_terminate(process)


def mp_worker(args, pipe_exp, worker_id, learner_id=0):
    args.random_seed += worker_id + learner_id * args.worker_num
    args.init_before_training(process_id=-1)

    if True:
        '''arguments: basic'''
        # cwd = args.cwd
        env = args.env
        agent = args.agent
        # gpu_id = args.gpu_id
        # worker_num = args.worker_num

        '''arguments: train'''
        net_dim = args.net_dim
        # max_memo = args.max_memo
        # break_step = args.break_step
        # batch_size = args.batch_size
        target_step = args.target_step
        # repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        # if_break_early = args.if_allow_break

        gamma = args.gamma
        reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        # soft_update_tau = args.soft_update_tau

        '''arguments: evaluate'''
        # show_gap = args.eval_gap
        # eval_times1 = args.eval_times1
        # eval_times2 = args.eval_times2
        # env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval

        '''arguments: environment'''
        # max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        # if_discrete = env.if_discrete
        del args  # In order to show these hyper-parameters clearly, I put them above.

    env.device = torch.device(f'cuda:{learner_id}')
    env_num = getattr(env, 'env_num', 0)
    '''init: Agent, ReplayBuffer, Evaluator'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, learner_id)
    # agent.act.eval()
    # [setattr(param, 'requires_grad', False) for param in agent.act.parameters()]

    if_on_policy = agent.if_on_policy

    if env_num:
        agent.state = env.reset_vec()
        agent.env_tensors = [[torch.zeros(0, dtype=torch.float32, device=agent.device)
                              for _ in range(5)]
                             for _ in range(env.env_num)]
        # 5 == len(states, actions, r_sums, logprobs, advantages)

        with torch.no_grad():
            while True:
                # pipe_exp[1].send((agent.act.state_dict(), agent.cri_target.state_dict()))
                act_state_dict, cri_target_state_dict = pipe_exp[0].recv()

                agent.act.load_state_dict(act_state_dict)
                agent.cri_target.load_state_dict(cri_target_state_dict)

                buffer = agent.explore_envs(env, target_step, reward_scale, gamma)
                buffer_tuple = agent.prepare_buffers(buffer)

                pipe_exp[0].send(buffer_tuple)
                # buffer_tuple = pipe_exp[1].recv()

    agent.state = env.reset()
    with torch.no_grad():
        while True:
            # pipe_exp[1].send((act_state_dict, cri_target_state_dict))
            act_state_dict, cri_target_state_dict = pipe_exp[0].recv()

            agent.act.load_state_dict(act_state_dict)
            agent.cri_target.load_state_dict(cri_target_state_dict)

            if if_on_policy:
                buffer = agent.explore_env(env, target_step, reward_scale, gamma)
                buffer_tuple = agent.prepare_buffer(buffer)  # buffer_tuple
            else:
                trajectory_list = agent.explore_env(env, target_step, reward_scale, gamma)
                state = torch.as_tensor([item[0] for item in trajectory_list], dtype=torch.float32, device=agent.device)
                other = torch.as_tensor([item[1] for item in trajectory_list], dtype=torch.float32, device=agent.device)
                buffer_tuple = (state, other)

            pipe_exp[0].send(buffer_tuple)
            # buffer_tuple = pipe_exp[1].recv()


def mp_learner(args, pipe_eva, pipe_exp_list, pipe_net_list=None, learner_id=0):
    args.init_before_training(process_id=learner_id)

    if True:
        '''arguments: basic'''
        # cwd = args.cwd
        env = args.env
        agent = args.agent
        # gpu_id = args.gpu_id
        worker_num = args.worker_num

        '''arguments: train'''
        net_dim = args.net_dim
        max_memo = args.max_memo
        # break_step = args.break_step
        batch_size = args.batch_size
        target_step = args.target_step
        repeat_times = args.repeat_times
        learning_rate = args.learning_rate
        # if_break_early = args.if_allow_break

        gamma = args.gamma
        reward_scale = args.reward_scale
        if_per_or_gae = args.if_per_or_gae
        soft_update_tau = args.soft_update_tau

        '''arguments: evaluate'''
        # show_gap = args.eval_gap
        # eval_times1 = args.eval_times1
        # eval_times2 = args.eval_times2
        # env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval

        '''arguments: environment'''
        # max_step = env.max_step
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete
        del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: Comm'''
    comm = LearnerComm(pipe_net_list, learner_id) if pipe_net_list is not None else None

    '''init: Agent'''
    agent.init(net_dim, state_dim, action_dim, learning_rate, if_per_or_gae, learner_id)
    if_on_policy = agent.if_on_policy

    '''init: ReplayBuffer'''
    if if_on_policy:
        steps = 0
        buffer = list()
    else:  # explore_before_training for off-policy
        buffer = ReplayBufferMP(max_len=target_step if if_on_policy else max_memo, worker_num=worker_num,
                                if_on_policy=if_on_policy, if_per_or_gae=if_per_or_gae,
                                state_dim=state_dim, action_dim=action_dim, if_discrete=if_discrete, )

        with torch.no_grad():  # update replay buffer
            trajectory_list = explore_before_training(env, target_step, reward_scale, gamma)
            steps = len(trajectory_list)

            buffer.buffers[0].extend_buffer_from_list(trajectory_list)
        agent.update_net(buffer, target_step, batch_size, repeat_times)  # pre-training and hard update

        # hard update for the first time
        agent.act_target.load_state_dict(agent.act.state_dict()) if getattr(agent, 'act_target', None) else None
        agent.cri_target.load_state_dict(agent.cri.state_dict()) if getattr(agent, 'cri_target', None) else None

    '''init: Evaluator'''
    # evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
    #                       eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    act_cpu = deepcopy(agent.act).to(torch.device("cpu"))  # for pipe1_eva
    act_cpu.eval()
    [setattr(param, 'requires_grad', False) for param in act_cpu.parameters()]
    if learner_id == 0:
        pipe_eva[1].send((act_cpu, steps))
        # act_cpu, steps = pipe_eva[0].recv()

    '''start training'''
    if_train = True
    while if_train:
        '''explore'''
        for pipe_exp in pipe_exp_list:
            pipe_exp[1].send((agent.act.state_dict(), agent.cri_target.state_dict()))
            # act_state_dict, cri_target_state_dict = pipe_exp[0].recv()

        steps = 0
        r_exp = 0
        buffer_tuples = list()
        for pipe_exp in pipe_exp_list:
            # pipe_exp[0].send(buffer_tuple)
            buffer_tuples.append(pipe_exp[1].recv())

        if if_on_policy:
            buffer = list()
            for buffer_tuple in buffer_tuples:
                steps += buffer_tuple[2].size(0)  # buffer[2] = r_sum
                r_exp += buffer_tuple[2].mean().item()  # buffer[2] = r_sum
                buffer.append(buffer_tuple)

            if (buffer_tuples is not None) and (comm is not None):
                buffer_tuples = comm.comm(buffer_tuples, 0, if_cuda=True)
                for buffer_tuple in buffer_tuples:
                    buffer.append(buffer_tuple)
        else:
            for i in range(worker_num):
                state, other = buffer_tuples[i]
                steps += other.size()[0]
                r_exp += other[:, 0].mean().item()  # other = (reward, mask, ...)
                buffer.buffers[i].extend_buffer(state, other)

            if comm is not None:
                buffer_tuples = comm.comm(buffer_tuples, 0)
                for i in range(worker_num):
                    state, other = buffer_tuples[i]
                    buffer.buffers[i].extend_buffer(state, other)

        logging_tuple = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
        if comm is not None:
            for round_id in range(comm.round_num):
                data = agent.act, agent.cri, agent.act_optim, agent.cri_optim,
                data = comm.comm(data, round_id)

                if data is None:
                    continue
                # isinstance(data, tuple):
                avg_update_net(agent.act, data[0], agent.device)
                avg_update_net(agent.cri, data[1], agent.device)
                avg_update_optim(agent.act_optim, data[2], agent.device)
                avg_update_optim(agent.cri_optim, data[3], agent.device)

        '''evaluate'''
        if learner_id == 0:
            if not pipe_eva[0].poll():
                act_cpu.load_state_dict(agent.act.state_dict())
                act_state_dict = act_cpu.state_dict()
            else:
                act_state_dict = None
            pipe_eva[1].send((act_state_dict, steps, r_exp, logging_tuple))
            # act_state_dict, steps, r_exp, logging_tuple = pipe_eva[0].recv()

        if pipe_eva[1].poll():
            # pipe_eva[0].send(if_train)
            if_train = pipe_eva[1].recv()

    comm.close_comm() if comm is not None else None
    empty_pipe_list(pipe_eva)
    for pipe_exp in pipe_exp_list:
        empty_pipe_list(pipe_exp)


def mp_evaluator(args, pipe_eva, learner_id=0):
    args.init_before_training(process_id=-1)

    if True:
        '''arguments: basic'''
        cwd = args.cwd
        env = args.env
        agent = args.agent
        gpu_id = args.gpu_id
        # worker_num = args.worker_num

        '''arguments: train'''
        # net_dim = args.net_dim
        # max_memo = args.max_memo
        break_step = args.break_step
        # batch_size = args.batch_size
        # target_step = args.target_step
        # repeat_times = args.repeat_times
        # learning_rate = args.learning_rate
        if_break_early = args.if_allow_break

        # gamma = args.gamma
        # reward_scale = args.reward_scale
        # if_per_or_gae = args.if_per_or_gae
        # soft_update_tau = args.soft_update_tau

        '''arguments: evaluate'''
        show_gap = args.eval_gap
        eval_times1 = args.eval_times1
        eval_times2 = args.eval_times2
        env_eval = deepcopy_or_rebuild_env(env) if args.env_eval is None else args.env_eval

        '''arguments: environment'''
        # max_step = env.max_step
        # state_dim = env.state_dim
        # action_dim = env.action_dim
        # if_discrete = env.if_discrete
        del args  # In order to show these hyper-parameters clearly, I put them above.

    '''init: Evaluator'''
    learner_num = 1 if isinstance(gpu_id, int) else len(gpu_id)
    gpu_id = gpu_id if isinstance(gpu_id, int) else gpu_id[learner_id]
    evaluator = Evaluator(cwd=cwd, agent_id=gpu_id, device=agent.device, env=env_eval,
                          eval_times1=eval_times1, eval_times2=eval_times2, eval_gap=show_gap)  # build Evaluator
    evaluator.eval_time += learner_id * (show_gap / learner_num)

    # pipe_eva[1].send((act_cpu, steps))
    act_cpu, steps = pipe_eva[0].recv()

    '''start training'''
    sum_step = steps
    if_train = True
    while if_train:
        # pipe_eva[1].send((act_state_dict, steps, r_exp, logging_tuple))
        act_state_dict, steps, r_exp, logging_tuple = pipe_eva[0].recv()

        sum_step += steps
        if act_state_dict is not None:
            act_cpu.load_state_dict(act_state_dict)

            if_reach_goal = evaluator.evaluate_and_save(act_cpu, sum_step, r_exp, logging_tuple)
            sum_step = 0

            if_train = not ((if_break_early and if_reach_goal)
                            or evaluator.total_step > break_step
                            or os.path.exists(f'{cwd}/stop'))

    print(f'| SavedDir: {cwd}\n'
          f'| UsedTime: {time.time() - evaluator.start_time:.0f}')
    pipe_eva[0].send(if_train)
    # if_train = pipe_eva[1].recv()


def process_safely_terminate(process):
    for p in process:
        try:
            p.terminate()
        except OSError as e:
            print(e)
            pass


def empty_pipe_list(pipe_list):
    for pipe in pipe_list:
        try:
            while pipe.poll():
                pipe.recv()
        except EOFError:
            pass


'''multiple GPU training'''


def train_and_evaluate_mg(args):  # multiple GPU
    import multiprocessing as mp  # Python built-in multiprocessing library
    process = list()

    pipe_net_list = [mp.Pipe() for _ in args.gpu_id]
    pipe_eva_list = [mp.Pipe() for _ in args.gpu_id]

    for learner_id in range(len(args.gpu_id)):
        pipe_exp_list = [mp.Pipe() for _ in range(args.worker_num)]
        pipe_eva = pipe_eva_list[learner_id]

        process.append(mp.Process(target=mp_learner, args=(args, pipe_eva, pipe_exp_list, pipe_net_list, learner_id)))
        process.extend([mp.Process(target=mp_worker, args=(args, pipe_exp_list[worker_id], worker_id, learner_id))
                        for worker_id in range(args.worker_num)])

    learner_id = 0
    pipe_eva = pipe_eva_list[learner_id]
    process.append(mp.Process(target=mp_evaluator, args=(args, pipe_eva, learner_id)))

    [p.start() for p in process]
    process[0].join()  # wait
    for learner_id in range(len(args.gpu_id)):
        pipe_eva = pipe_eva_list[learner_id]
        pipe_net = pipe_net_list[learner_id]

        pipe_eva[0].send(False)
        [pipe_net[0].send(None) for _ in range(3)]
    process_safely_terminate(process[1:])


class LearnerComm:
    def __init__(self, pipe_net_list, learner_id):
        pipe_num = len(pipe_net_list)
        self.pipe_net_list = pipe_net_list
        self.device_list = [torch.device(f'cuda:{i}') for i in range(pipe_num)]

        if pipe_num == 2:
            self.round_num = 1
            if learner_id == 0:
                self.pipe0 = pipe_net_list[0]
                self.idx_l = (1,)
            else:  # if learner_id == 1:
                self.pipe0 = pipe_net_list[1]
                self.idx_l = (0,)
        else:  # if pipe_num == 4:
            self.round_num = 1
            if learner_id == 0:
                self.pipe0 = pipe_net_list[learner_id]
                self.idx_l = (1, 2)
            elif learner_id == 1:
                self.pipe0 = pipe_net_list[learner_id]
                self.idx_l = (0, 3)
            elif learner_id == 2:
                self.pipe0 = pipe_net_list[learner_id]
                self.idx_l = (3, 1)
            else:  # if learner_id == 3:
                self.pipe0 = pipe_net_list[learner_id]
                self.idx_l = (2, 1)

    def comm(self, data, round_id, if_cuda=False):
        idx = self.idx_l[round_id]

        if if_cuda:
            data = [[t.to(self.device_list[idx]) for t in item]
                    for item in data]

        self.pipe_net_list[idx][0].send(data)
        return self.pipe0[1].recv()

    def close_comm(self):
        for pipe_net in self.pipe_net_list:
            empty_pipe_list(pipe_net)


def get_optim_parameters(optim):  # for avg_update_optim()
    params_list = list()
    for params_dict in optim.state_dict()['state'].values():
        params_list.extend([t for t in params_dict.values() if isinstance(t, torch.Tensor)])
    return params_list


def avg_update_optim(dst_optim, src_optim, device):
    for dst, src in zip(get_optim_parameters(dst_optim),
                        get_optim_parameters(src_optim)):
        dst.data.copy_((dst.data + src.data.to(device)) * 0.5)
        # dst.data.copy_(src.data * tau + dst.data * (1 - tau))


def avg_update_net(dst_net, src_net, device):
    for tar, cur in zip(dst_net.parameters(), src_net.parameters()):
        tar.data.copy_((tar.data + cur.data.to(device)) * 0.5)


'''utils'''


class Evaluator:
    def __init__(self, cwd, agent_id, eval_times1, eval_times2, eval_gap, env, device):
        self.recorder = list()  # total_step, r_avg, r_std, obj_c, ...
        self.r_max = -np.inf
        self.total_step = 0

        self.env = env
        self.cwd = cwd
        self.device = device
        self.agent_id = agent_id
        self.eval_gap = eval_gap
        self.eval_times1 = eval_times1
        self.eval_times2 = eval_times2
        self.target_return = env.target_return

        self.used_time = None
        self.start_time = time.time()
        self.eval_time = 0
        print(f"{'#' * 80}\n"
              f"{'Step':>8}{'maxR':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'objC':>8}{'etc.':>8}")

    def evaluate_and_save(self, act, steps, r_exp, log_tuple) -> bool:
        self.total_step += steps  # update total training steps

        if time.time() - self.eval_time < self.eval_gap:
            return False  # if_reach_goal

        self.eval_time = time.time()
        rewards_steps_list = [get_episode_return_and_step(self.env, act, self.device) for _ in
                              range(self.eval_times1)]
        r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)

        if r_avg > self.r_max:  # evaluate actor twice to save CPU Usage and keep precision
            rewards_steps_list += [get_episode_return_and_step(self.env, act, self.device)
                                   for _ in range(self.eval_times2 - self.eval_times1)]
            r_avg, r_std, s_avg, s_std = self.get_r_avg_std_s_avg_std(rewards_steps_list)
        if r_avg > self.r_max:  # save checkpoint with highest episode return
            self.r_max = r_avg  # update max reward (episode return)

            act_save_path = f'{self.cwd}/actor.pth'
            torch.save(act.state_dict(), act_save_path)  # save policy network in *.pth
            print(f"{self.total_step:8.2e}{self.r_max:8.2f} |")  # save policy and print

        self.recorder.append((self.total_step, r_avg, r_std, *log_tuple))  # update recorder

        if_reach_goal = bool(self.r_max > self.target_return)  # check if_reach_goal
        if if_reach_goal and self.used_time is None:
            self.used_time = int(time.time() - self.start_time)
            print(f"{'Step':>8}{'TargetR':>8} |"
                  f"{'avgR':>8}{'stdR':>7} |{'UsedTime':>8}  ########\n"
                  f"{self.total_step:8.2e}{self.target_return:8.2f} |"
                  f"{r_avg:8.2f}{r_std:7.1f} |{self.used_time:>8}  ########")

        print(f"{self.total_step:8.2e}{self.r_max:8.2f} |"
              f"{r_avg:8.2f}{r_std:7.1f}{s_avg:7.0f}{s_std:6.0f} |"
              f"{r_exp:8.2f}{''.join(f'{n:8.2f}' for n in log_tuple)}")
        self.draw_plot()
        return if_reach_goal

    def draw_plot(self):
        if len(self.recorder) == 0:
            print("| save_npy_draw_plot() WARNNING: len(self.recorder)==0")
            return None

        '''convert to array and save as npy'''
        np.save('%s/recorder.npy' % self.cwd, self.recorder)

        '''draw plot and save as png'''
        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        save_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.r_max:.3f}"

        save_learning_curve(self.recorder, self.cwd, save_title)

    @staticmethod
    def get_r_avg_std_s_avg_std(rewards_steps_list):
        rewards_steps_ary = np.array(rewards_steps_list, dtype=np.float)
        r_avg, s_avg = rewards_steps_ary.mean(axis=0)  # average of episode return and episode step
        r_std, s_std = rewards_steps_ary.std(axis=0)  # standard dev. of episode return and episode step
        return r_avg, r_std, s_avg, s_std


def get_episode_return_and_step(env, act, device) -> (float, int):
    episode_return = 0.0  # sum of rewards in an episode
    episode_step = 1
    max_step = env.max_step
    if_discrete = env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step


def save_learning_curve(recorder, cwd='.', save_title='learning curve', fig_name='plot_learning_curve.jpg'):
    recorder = np.array(recorder)  # recorder_ary.append((self.total_step, r_avg, r_std, obj_a, obj_c))
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    obj_c = recorder[:, 3]
    obj_a = recorder[:, 4]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    axs0 = axs[0]
    axs0.cla()
    color0 = 'lightcoral'
    axs0.set_xlabel('Total Steps')
    axs0.set_ylabel('Episode Return')
    axs0.plot(steps, r_avg, label='Episode Return', color=color0)
    axs0.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)

    ax11 = axs[1]
    ax11.cla()
    color11 = 'royalblue'
    axs0.set_xlabel('Total Steps')
    ax11.set_ylabel('objA', color=color11)
    ax11.plot(steps, obj_a, label='objA', color=color11)
    ax11.tick_params(axis='y', labelcolor=color11)
    for plot_i in range(5, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax11.plot(steps, other, label=f'{plot_i}', color='grey')

    ax12 = axs[1].twinx()
    color12 = 'darkcyan'
    ax12.set_ylabel('objC', color=color12)
    ax12.fill_between(steps, obj_c, facecolor=color12, alpha=0.2, )
    ax12.tick_params(axis='y', labelcolor=color12)

    '''plot save'''
    plt.title(save_title, y=2.3)
    plt.savefig(f"{cwd}/{fig_name}")
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


def explore_before_training(env, target_step, reward_scale, gamma) -> (list, np.ndarray):  # for off-policy only
    trajectory_list = list()

    if_discrete = env.if_discrete
    action_dim = env.action_dim

    state = env.reset()
    step = 0
    while True:
        if if_discrete:
            action = rd.randint(action_dim)  # assert isinstance(action_int)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, action)
        else:
            action = rd.uniform(-1, 1, size=action_dim)
            next_s, reward, done, _ = env.step(action)
            other = (reward * reward_scale, 0.0 if done else gamma, *action)

        trajectory_list.append((state, other))
        state = env.reset() if done else next_s

        step += 1
        if done and step > target_step:
            break
    return trajectory_list
