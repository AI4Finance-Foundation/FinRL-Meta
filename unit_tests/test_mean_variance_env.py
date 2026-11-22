import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from meta.env_mean_variance import theory
from meta.env_mean_variance.emv_agent import EMVAgent
from meta.env_mean_variance.emv_agent import EMVConfig
from meta.env_mean_variance.env_mean_variance import MeanVarianceEnv


def test_env_follows_the_gymnasium_api():
    from gymnasium.utils.env_checker import check_env

    check_env(MeanVarianceEnv(), skip_render_check=True)


def test_env_is_reproducible_and_rewards_only_at_the_end():
    def rollout():
        env = MeanVarianceEnv(dt=0.1)
        obs, _ = env.reset(seed=3)
        rewards, path = [], [obs[1]]
        terminated = False
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(np.array([0.5]))
            rewards.append(reward)
            path.append(obs[1])
            assert not truncated
            assert info["wealth"] == obs[1]
        return np.array(rewards), np.array(path), env

    rewards, path, env = rollout()
    rewards2, path2, _ = rollout()
    np.testing.assert_array_equal(path, path2)
    assert len(rewards) == env.n_steps == 10
    assert np.all(rewards[:-1] == 0)
    assert rewards[-1] == pytest.approx(-((path[-1] - env.w) ** 2))


def test_reset_sets_the_lagrange_multiplier():
    env = MeanVarianceEnv(z=1.4)
    assert env.w == 1.4
    env.reset(seed=0, options={"w": 1.7})
    assert env.w == 1.7


def test_no_risky_holding_keeps_wealth_constant():
    env = MeanVarianceEnv(dt=0.1)
    obs, _ = env.reset(seed=0)
    terminated = False
    while not terminated:
        obs, _, terminated, _, _ = env.step(np.array([0.0]))
    assert obs[1] == env.x0


def test_classical_optimal_policy_reaches_the_efficient_frontier():
    # Under u = -rho/sigma (x - w*), E[x_T] = z and Var[x_T] = (z - x0)^2 / (e^{rho^2 T} - 1).
    # A low Sharpe ratio keeps the tails of x_T light, so the sample variance is reliable.
    env = MeanVarianceEnv(mu=0.1, sigma=0.2, dt=1.0 / 50)
    w = theory.optimal_lagrange_multiplier(env.z, env.x0, env.rho, env.T)
    n = 4000
    terminal = np.empty(n)
    obs, _ = env.reset(seed=0, options={"w": w})
    for k in range(n):
        if k > 0:
            obs, _ = env.reset()
        terminated = False
        while not terminated:
            u = theory.classical_policy(obs[1], w, env.rho, env.sigma)
            obs, _, terminated, _, _ = env.step(np.array([u]))
        terminal[k] = obs[1]
    mean, var = theory.optimal_terminal_moments(env.z, env.x0, env.rho, env.T)
    assert terminal.mean() == pytest.approx(mean, abs=5 * terminal.std() / np.sqrt(n))
    # the Euler scheme with 50 steps slightly changes the variance
    assert terminal.var() == pytest.approx(var, rel=0.1)


def test_emv_gradients_match_finite_differences():
    agent = EMVAgent(MeanVarianceEnv())
    agent.theta1, agent.theta2, agent.phi1, agent.phi2, agent.w = (
        0.3,
        -0.2,
        0.1,
        0.8,
        1.3,
    )
    x = agent.run_episode(np.random.default_rng(0), seed=1)
    analytic = agent.gradients(x)
    eps = 1e-6
    for name in ("theta1", "theta2", "phi1", "phi2"):
        original = getattr(agent, name)
        setattr(agent, name, original + eps)
        up = agent.loss(x)
        setattr(agent, name, original - eps)
        down = agent.loss(x)
        setattr(agent, name, original)
        assert analytic[name] == pytest.approx(
            (up - down) / (2 * eps), rel=1e-5, abs=1e-8
        ), name


@pytest.mark.parametrize(("mu", "sigma"), [(0.3, 0.1), (0.1, 0.2), (0.5, 0.4)])
def test_emv_policy_equals_the_optimal_policy_at_the_true_parameters(mu, sigma):
    # Eq. (45): 2 phi2 = rho^2 and sigma^2 = lam pi e^{1 - 2 phi1}.
    env = MeanVarianceEnv(mu=mu, sigma=sigma)
    agent = EMVAgent(env)
    lam = agent.config.lam
    agent.phi2 = env.rho**2 / 2
    agent.phi1 = (1.0 - np.log(sigma**2 / (lam * np.pi))) / 2
    for t, x in [(0.0, 1.0), (0.5, 1.2), (0.9, 0.8)]:
        expected = theory.exploratory_policy(t, x, agent.w, env.rho, sigma, lam, env.T)
        assert agent.policy(t, x) == pytest.approx(expected, rel=1e-12)


def test_lagrange_multiplier_moves_toward_the_target():
    agent = EMVAgent(MeanVarianceEnv(z=1.4), EMVConfig(N=2, alpha=0.1, init_w=1.5))
    agent.update_lagrange_multiplier(1.6)
    assert agent.w == 1.5  # waits for N terminal wealths
    agent.update_lagrange_multiplier(1.8)
    assert agent.w == pytest.approx(1.5 - 0.1 * (1.7 - 1.4))  # too much wealth


def test_emv_training_is_reproducible():
    def run():
        return EMVAgent(MeanVarianceEnv()).train(20, seed=3)

    np.testing.assert_array_equal(run(), run())


def test_sign_aware_emv_learns_to_short_a_falling_market():
    env = MeanVarianceEnv(mu=-0.3, sigma=0.1)
    agent = EMVAgent(env, EMVConfig(sign_aware=True))
    agent.train(50, seed=0)
    assert agent.sign == -1.0
    # below w, the optimal policy (24) shorts a falling stock: -rho / sigma (x - w) < 0
    x = agent.w - 0.3
    assert agent.policy(0.3, x)[0] < 0
    assert theory.classical_policy(x, agent.w, env.rho, env.sigma) < 0
    # without sign awareness, policy (46) can only hold the stock long below w
    long_only = EMVAgent(env)
    assert long_only.sign == 1.0
    assert long_only.policy(0.3, long_only.w - 0.3)[0] > 0


def test_gradient_clipping_bounds_each_update():
    config = EMVConfig(grad_clip=0.5)
    agent = EMVAgent(MeanVarianceEnv(mu=0.1, sigma=0.2), config)
    agent.phi2, agent.w = 2.0, 3.0  # large TD errors
    x = agent.run_episode(np.random.default_rng(0), seed=1)
    assert max(abs(v) for v in agent.gradients(x).values()) > config.grad_clip
    before = (agent.theta1, agent.theta2, agent.phi1, agent.phi2)
    agent.update(x)
    after = (agent.theta1, agent.theta2, agent.phi1, agent.phi2)
    steps = [config.eta_theta, config.eta_theta, config.eta_phi, config.eta_phi]
    for b, a, eta in zip(before, after, steps):
        assert abs(a - b) <= eta * config.grad_clip + 1e-15


def test_agent_and_market_noise_are_independent():
    # gymnasium seeds the env like np.random.default_rng(seed); if the agent used the
    # same seed, it would always buy right before the price rises.
    terminal = EMVAgent(MeanVarianceEnv(mu=0.3, sigma=0.2)).train(50, seed=0)
    # the initial policy barely trades, so wealth should stay close to x0 = 1
    assert abs(terminal.mean() - 1.0) < 0.05
