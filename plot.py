def draw_cumulative_return(state_dim, action_dim, env, args, _torch) -> list:

    agent = args.agent
    net_dim = args.net_dim
    cwd = args.cwd

    agent.init(net_dim, state_dim, action_dim)
    agent.save_load_model(cwd=cwd, if_save=False)
    act = agent.act
    device = agent.device

    state = env.reset()
    episode_returns = list()
    episode_returns.append(1)
    with _torch.no_grad():
        for i in range(env.max_step):
            if i == 0:
                init_price = env.day_price[0]
            s_tensor = _torch.as_tensor((state,), device=device)
            a_tensor = act(s_tensor)  # action_tanh = act.forward()
            action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
            state, reward, done, _ = env.step(action)
                
            episode_returns.append(env.total_asset/1e6)
            if done:
                break

    import matplotlib.pyplot as plt
    plt.plot(episode_returns, label='agent return')
    plt.grid()
    plt.title('cumulative return')
    plt.xlabel('time')
    plt.legend()
    plt.savefig(f'{cwd}/cumulative_return.jpg')
    return episode_returns
