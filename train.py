from agents.elegantrl_models import DRLAgent as DRLAgent_erl
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3
from finrl_meta.data_processor import DataProcessor


def train(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, model_name, if_vix=True,
          **kwargs):
    # process data using unified data processor
    dp = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = dp.run(ticker_list, 
                                                       technical_indicator_list,
                                                       if_vix)
    data_config = {'price_array': price_array,
                   'tech_array': tech_array,
                   'turbulence_array': turbulence_array}
    # build environment using processed data
    env_instance = env(config=data_config)

    # read parameters and load agents
    cwd = kwargs.get('cwd', './' + str(model_name))  # cwd: current_working_dir

    if drl_lib == 'elegantrl':
        break_step = kwargs.get('break_step', 1e6)  # total_training_steps
        erl_params = kwargs.get('erl_params')  # see notebooks for examples.

        agent = DRLAgent_erl(env=env,
                             price_array=price_array,
                             tech_array=tech_array,
                             turbulence_array=turbulence_array)

        model = agent.get_model(model_name, model_kwargs=erl_params)
        trained_model = agent.train_model(model=model,
                                          cwd=cwd,
                                          total_timesteps=break_step)  # erl model is automated saved in cwd

    elif drl_lib == 'rllib':
        total_episodes = kwargs.get('total_episodes', 100)  # rllib uses total training episodes instead of steps.
        rllib_params = kwargs.get('rllib_params')

        agent_rllib = DRLAgent_rllib(env=env,
                                     price_array=price_array,
                                     tech_array=tech_array,
                                     turbulence_array=turbulence_array)

        model, model_config = agent_rllib.get_model(model_name)

        model_config['lr'] = rllib_params['lr']  # learning_rate
        model_config['train_batch_size'] = rllib_params['train_batch_size']
        model_config['gamma'] = rllib_params['gamma']

        trained_model = agent_rllib.train_model(model=model,
                                                model_name=model_name,
                                                model_config=model_config,
                                                total_episodes=total_episodes)
        trained_model.save(cwd)


    elif drl_lib == 'stable_baselines3':
        total_timesteps = kwargs.get('total_timesteps', 1e6)
        agent_params = kwargs.get('agent_params')

        agent = DRLAgent_sb3(env=env_instance)

        model = agent.get_model(model_name, model_kwargs=agent_params)
        trained_model = agent.train_model(model=model,
                                          tb_log_name=model_name,
                                          total_timesteps=total_timesteps)
        print('Training finished!')
        trained_model.save(cwd)
        print('Trained model saved in ' + str(cwd))
    else:
        raise ValueError('DRL library input is NOT supported. Please check.')
