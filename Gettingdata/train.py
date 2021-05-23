from video import VideoRecorderCallback
import gym


from stable_baselines3 import PPO

env = gym.make('CartPole-v0')
model = PPO("MlpPolicy", env, tensorboard_log="./data_collection/", learning_rate=0.0051, verbose=1)

video_recorder = VideoRecorderCallback(env, render_freq=10000)

model.learn(total_timesteps=int(50000), callback=video_recorder, tb_log_name='Learning_rate=0.0051_PPO')

