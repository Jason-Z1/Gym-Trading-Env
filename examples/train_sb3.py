"""Minimal training example using Stable-Baselines3 PPO with the TradingEnv.

This script demonstrates a modern RL pipeline: vectorized envs, observation flattening,
reward normalization and simple model training. Requires `stable-baselines3` and `sb3-contrib`.
"""
import sys
sys.path.append("./src")

import gymnasium as gym
from gym_trading_env.wrappers import FlattenObservation, RewardNormalizer
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except Exception as e:
    raise SystemExit("Install stable-baselines3 to run this example: pip install stable-baselines3")

import pandas as pd


def make_env():
    df = pd.read_csv("examples/data/BTC_USD-Hourly.csv", parse_dates=["date"], index_col="date")
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7 * 24).max()
    df.dropna(inplace=True)

    env = gym.make(
        "TradingEnv",
        name="BTCUSD",
        df=df,
        windows=5,
        positions=[-1, -0.5, 0, 0.5, 1],
        trading_fees=0.01 / 100,
        borrow_interest_rate=0.0003 / 100,
        reward_function=lambda history: np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]),
        portfolio_initial_value=1000,
        max_episode_duration=500,
        disable_env_checker=True,
    )

    # Wrap: flatten observations and normalize rewards
    env = FlattenObservation(env)
    env = RewardNormalizer(env)
    return env


if __name__ == "__main__":
    # vectorized environments
    n_envs = 4
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])
    # optionally normalize observations and rewards across envs
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=10_000)
    model.save("ppo_trading")

    print("Training complete — model saved as ppo_trading.zip")
