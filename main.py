import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from env.StockTradingEnv import StockTradingEnv

import pandas as pd

df = pd.read_csv('SBI_train.csv')
df = df.sort_values('Date')
df=df.rename(columns={"Adj Close": "Adjusted_Close"})

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)

obs = env.reset()
# df = pd.read_csv('SBI_train.csv')
# df = df.sort_values('Date')
# df=df.rename(columns={"Adj Close": "Adjusted_Close"})

# # The algorithms require a vectorized environment to run
# env = DummyVecEnv([lambda: StockTradingEnv(df)])

for i in range(len(df['Date'])):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render(title="MSFT")
