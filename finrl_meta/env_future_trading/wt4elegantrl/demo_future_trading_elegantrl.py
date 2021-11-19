import torch
import numpy as np
from click import command, group, option
# from elegantrl.agent import AgentPPO as Agent
# from elegantrl.agent import AgentSAC as Agent
# from elegantrl.agent import AgentModSAC as Agent
from elegantrl.agent import AgentTD3 as Agent


# from elegantrl.agent import AgentDoubleDQN as Agent
# from elegantrl.agent import AgentDQN as Agent


from elegantrl.run import Arguments, train_and_evaluate
from envs_simple_cta import SimpleCTASubProcessEnv, SimpleCTAEnv
from gym import make, register
from numpy import inf
from os import getpid


class Wt4RLSimpleTrainer(SimpleCTASubProcessEnv):
    env_num = 1
    max_step = 1500
    if_discrete = False

    @property
    def state_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        # if len(self.action_space.shape) > 0 else 10
        return self.action_space.shape[0]

    def __init__(self):
        super().__init__(**{
            # 'time_start': 202108301600,
            # 'time_end': 202108311600,
            'time_range': (
                # (201901011600, 202101011600),

                # (201901011600, 201906301600),
                # (201906301600, 202001011600),
                # (202001011600, 202006301600),
                # (202006301600, 202101011600),

                #(201812311600, 201901311600),
                #(201901311600, 201902311600),
                #(201902311600, 201903311600),
                #(201903311600, 201904311600),
                #(201904311600, 201905311600),
                #(201905311600, 201906311600),
                #(201906311600, 201907311600),
                #(201907311600, 201908311600),
                #(201908311600, 201909311600),
                #(201909311600, 201910311600),
                #(201910311600, 201911311600),
                #(201911311600, 201912311600),

                #(201912311600, 202001311600),
                #(202001311600, 202002311600),
                #(202002311600, 202003311600),
                #(202003311600, 202004311600),
                #(202004311600, 202005311600),
                #(202005311600, 202006311600),
                #(202006311600, 202007311600),
                #(202007311600, 202008311600),
                #(202008311600, 202009311600),
                (202009311600, 202010311600),
                (202010311600, 202011311600),
                (202011311600, 202012311600),
            ),
            'slippage': 0,
            'mode': 1
        })


class Wt4RLSimpleEvaluator(SimpleCTASubProcessEnv):
    env_num = 1
    max_step = 1500
    if_discrete = False

    @property
    def state_dim(self):
        return self.observation_space.shape[0]

    @property
    def action_dim(self):
        # if len(self.action_space.shape) > 0 else 10
        return self.action_space.shape[0]

    def __init__(self):  # mode=3可以打开详细调试模式
        super().__init__(**{
            'time_range': (
                # (202101011600, 202106301600),
                # (201701011600, 201706301600),
                # (201706301600, 201801011600),
                # (201801011600, 201806301600),
                # (201806301600, 201901011600),


                (202012311600, 202101311600),
                (202101311600, 202102311600),
                (202102311600, 202103311600),
                (202103311600, 202104311600),
                (202104311600, 202105311600),
                (202105311600, 202106311600),

                (201612311600, 201701311600),
                (201701311600, 201702311600),
                (201702311600, 201703311600),
                (201703311600, 201704311600),
                (201704311600, 201705311600),
                (201705311600, 201706311600),
                (201706311600, 201707311600),
                (201707311600, 201708311600),
                (201708311600, 201709311600),
                (201709311600, 201710311600),
                (201710311600, 201711311600),
                (201711311600, 201712311600),

                (201712311600, 201801311600),
                (201801311600, 201802311600),
                (201802311600, 201803311600),
                (201803311600, 201804311600),
                (201804311600, 201805311600),
                (201805311600, 201806311600),
                (201806311600, 201807311600),
                (201807311600, 201808311600),
                (201808311600, 201809311600),
                (201809311600, 201810311600),
                (201810311600, 201811311600),
                (201811311600, 201812311600),
            ),
            'slippage': 0,
            'mode': 1
        })

class Wt4RLSimpleEvaluator2(SimpleCTAEnv):
    env_num = 1
    max_step = 1500
    if_discrete = False
    
    @property
    def state_dim(self):
        return self.observation_space.shape[0]
    
    @property
    def action_dim(self):
        # if len(self.action_space.shape) > 0 else 10
        return self.action_space.shape[0]
    
    def __init__(self):  # mode=3可以打开详细调试模式
        super().__init__(**{
            'time_range': (
                # (202101011600, 202106301600),
                # (201701011600, 201706301600),
                # (201706301600, 201801011600),
                # (201801011600, 201806301600),
                # (201806301600, 201901011600),
    
                (202012311600, 202101311600),
            ),
            'slippage': 0,
            'mode': 2
        })

register('wt4rl-simplecta-trainer-v0', entry_point=Wt4RLSimpleTrainer)
register('wt4rl-simplecta-evaluator-v0', entry_point=Wt4RLSimpleEvaluator)


if __name__ == '__main__':
    @group()
    def run():
        pass

    @command()
    @option('--count', default=24)
    def debug(count):
        env: SimpleCTASubProcessEnv = make('wt4rl-simplecta-trainer-v0')
        print('状态空间', env.observation_space.shape)
        print('动作空间', env.action_space.shape)
        for i in range(1, int(count)+1):  # 模拟训练10次
            obs = env.reset()
            done = False
            n = 0
            while not done:
                action = env.action_space.sample()  # 模拟智能体产生动作
                obs, reward, done, info = env.step(action)
                n += 1
                # print('action:', action, 'obs:', obs, 'reward:', reward, 'done:', done)
            print('第%s次训练完成，执行%s步, 奖励%s, 盈亏%s。' % (i, n, reward, env.assets))
        env.close()

    @command()
    def train():
        args = Arguments(
            env='wt4rl-simplecta-trainer-v0',
            # env='wt4rl-simplecta-evaluator-v0',
            agent=Agent()
        )

        # args必须设置的参数
        #args.eval_env = 'wt4rl-simplecta-evaluator-v0'
        args.max_step = 3000
        args.state_dim = 39
        args.action_dim = 3
        args.if_discrete = False
        args.target_return = 5  # inf
        args.if_overwrite = False
        args.eval_times1 = 15  # 待查明：为啥td3的评估器结果完全一致
        args.eval_times2 = 30  # 待查明：为啥td3的评估器结果完全一致

        args.worker_num = 1  # 内存小的注意别爆内存
        args.break_step = 1e5
        args.if_allow_break = True

        #
        args.gamma = 0.96  # 8小时会跨过一次隔夜风险，既96个bar
        args.learning_rate = 1e-5
        # args.gamma = 0.1 ** (1/12/8) # 8小时会跨过一次隔夜风险，既96个bar
        # args.learning_rate = 1e-3  # N15:294  Y14:292
        args.eval_gap = 2 ** 8
        args.net_dim = 2 ** 6
        args.batch_size = args.net_dim * 2
        args.max_memo = 2 ** 20
        args.target_step = args.max_step * 2
        args.if_per_or_gae = True
        # args.agent.if_use_cri_target = True
        # args.agent.if_use_dueling = True

        args.env_num = 1
        args.learner_gpus = (0,)
        args.workers_gpus = args.learner_gpus
        args.eval_gpu_id = 0
        args.cwd = './outputs_bt/elegantrl/%s_%s_%s' % (
            args.agent.__class__.__name__, args.gamma, args.learning_rate)
        # args.repeat_times = 0.01

        #args.net_dim = 2**9
        # args.net_dim = 2 ** 8
        #args.max_memo = 2 ** 22
        # args.break_step = args.max_step*1000
        # args.batch_size = 2 ** 11  # args.net_dim * 2
        # args.repeat_times = 1.5

        # args.eval_gap = 2 ** 9
        # args.eval_times1 = 2 ** 2
        # args.eval_times2 = 2 ** 5

        # args.worker_num = 4
        # args.target_step = args.env.max_step * 1
        # train_and_evaluate(args)

        train_and_evaluate(args)
        
    @command()
    def test():
        environment = Wt4RLSimpleEvaluator2()
        agent = Agent()

        agent.init(net_dim=2 ** 6, state_dim=39, action_dim=3)
        agent.save_or_load_agent(
            cwd='./outputs_bt/elegantrl/AgentTD3_0.96_1e-05/best', if_save=False)
        act = agent.act
        device = agent.device

        _torch = torch
        state = environment.reset()
        with _torch.no_grad():
            for i in range(environment.max_step):
                s_tensor = _torch.as_tensor((state,), dtype=torch.float32, device=device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                print(environment.assets)
                state, reward, done, _ = environment.step(action)
                if done:
                    break
        print("Test Finished!")


    run.add_command(debug)
    run.add_command(train)
    run.add_command(test)
    # run.add_command(eval)

    run()
