import retro
from stable_baselines3 import A2C, PPO, SAC, DDPG, DQN
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

env = retro.make(game='SuperMarioBros-Nes')

#run without skip next
env = MaxAndSkipEnv(env, 4)
env = GrayScaleObservation(env, keep_dim=True)
path = './logs/PPO_CNN_00003lr_grey2//rl_model_1500000_steps'

model = PPO.load(path=path, env=env)

obs = env.reset()
run = False

while not run:
    action, state = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()