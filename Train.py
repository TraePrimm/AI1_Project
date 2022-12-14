import retro
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed



def main():
    env = retro.make(game='SuperMarioBros-Nes')
    #eval_env = retro.make(game = 'SuperMarioBros-Nes')
    # print(env.action_space.sample()) #[0 1 1 1 0 1 1 1 1 1 0 1]
    #print(env.observation_space.shape) #(224, 256, 3)
    #print(env.action_space.sample()) # [1 1 0 1 0 0 1 0 1 0 0 1]
    TimeSteps = 5_000_000
    set_random_seed(0)

    ########################
    # run this command to view tensorboards
    # tensorboard --logdir=tensorboards/
    #########################

    
    '''
    checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./logs/A2C/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )
    
    
    
    
   
    model = A2C("MlpPolicy",env, tensorboard_log="./tensorboards/")
    model.learn(total_timesteps=TimeSteps, callback=checkpoint_callback, reset_num_timesteps=False, tb_log_name="A2C")
    
    
    checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./logs/PPO/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )



    model = PPO("MlpPolicy",env,verbose=1, tensorboard_log="./tensorboards/")
    model.learn(total_timesteps=TimeSteps,callback = checkpoint_callback, reset_num_timesteps=False, tb_log_name="PPO")
    '''
    
    
    
    '''

    If I go back to retrain use 


    checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path="./logs/A2C_CNN_00003lr/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )



    model = A2C("CnnPolicy",env,verbose=1, tensorboard_log="./tensorboards/", learning_rate=.00003)
    path = './logs/A2C_CNN_00003lr/rl_model_3600000_steps'
    #model.load(path= path, env = env)
    
    model.learn(total_timesteps=TimeSteps,callback = checkpoint_callback, reset_num_timesteps=False, tb_log_name="A2C_CNN_00003lr")
    
    '''
    

    checkpoint_callback = CheckpointCallback(
    save_freq=200_000,
    save_path="./logs/PPO_CNN_00003lr/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
    )


    model = PPO("CnnPolicy",env,verbose=1, tensorboard_log="./tensorboards/", learning_rate=.00003)
    model.learn(total_timesteps=TimeSteps, callback = checkpoint_callback, reset_num_timesteps=False, tb_log_name="PPO_CNN_00003lr")
    


if __name__ == "__main__":
    main()