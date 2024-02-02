import os
import shutil
import torch.nn as nn
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from custom_callbacks import EnvDumpCallback, TensorboardCallback
from stable_baselines3 import SAC
import argparse
import gymnasium
from trainer import Trainer

# define constants
ENV_NAME = "Humanoid-v4"
# ENV_NAME = "Walker2d-Footstep"
# ENV_NAME = "Walker2d-v4"

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join("output", "training_sac", now) + ENV_NAME

# Path to normalized Vectorized environment and best model (if not first task)
PATH_TO_NORMALIZED_ENV = None
PATH_TO_PRETRAINED_NET = None

max_episode_steps = 1000  # default: 100

# Function that creates and monitors vectorized environments:
def make_parallel_envs(num_env, start_index=0):
    def make_env(_):
        def _thunk():
            env = gymnasium.make(ENV_NAME)
            env._max_episode_steps = max_episode_steps
            env = Monitor(env, TENSORBOARD_LOG)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()




    if args.train:
        print("Training", ENV_NAME, "with SAC")

        # ensure tensorboard log directory exists and copy this file to track
        os.makedirs(TENSORBOARD_LOG, exist_ok=True)
        shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

        envs = make_parallel_envs(16)
        envs = VecNormalize(envs)

        # Define callbacks for evaluation and saving the agent
        eval_callback = EvalCallback(
            eval_env=envs,
            callback_on_new_best=EnvDumpCallback(TENSORBOARD_LOG, verbose=0),
            n_eval_episodes=10,
            best_model_save_path=TENSORBOARD_LOG,
            log_path=TENSORBOARD_LOG,
            eval_freq=10_000,
            deterministic=True,
            render=False,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=25_000,
            save_path=TENSORBOARD_LOG,
            save_vecnormalize=True,
            verbose=1,
        )

        callbacks = [eval_callback, checkpoint_callback]

        model_config = dict(batch_size=32, learning_rate=2.55673e-05) # model config for humaniod

        model = SAC("MlpPolicy", envs, verbose=1, tensorboard_log=TENSORBOARD_LOG)
        model.learn(total_timesteps=1_000_000, callback=callbacks, log_interval=4)


        model.save(os.path.join(TENSORBOARD_LOG), "final_sac_humanoid.pkl")
        envs.save(os.path.join(TENSORBOARD_LOG), "final_sac_humanoid.pkl")
        print("Done. Model saved.")

    if args.eval:

        # # Evaluation 1 -- Walker Footstep (upright balancing)
        # net_path = "output/training_sac/2024-02-02/10-50-31Walker2d-Footstep/best_model.zip"
        # env_path = "output/training_sac/2024-02-02/10-50-31Walker2d-Footstep/training_env.pkl"

        # # Evaluation 2 -- Walker Footstep (learns to stand still)
        # net_path = "output/training_sac/2024-02-02/11-23-36Walker2d-Footstep/best_model.zip"
        # env_path = "output/training_sac/2024-02-02/11-23-36Walker2d-Footstep/training_env.pkl"

        # # Evaluation 3 -- Walker Footstep (upright balancing, shuffling)
        # net_path = "output/training_sac/2024-02-02/11-24-20Walker2d-Footstep/best_model.zip"
        # env_path = "output/training_sac/2024-02-02/11-24-20Walker2d-Footstep/training_env.pkl"


        # # Evaluation 1 -- Humanoid
        # net_path = "output/training_sac/2024-02-02/10-54-04Humanoid-v4/best_model.zip"
        # env_path = "output/training_sac/2024-02-02/10-54-04Humanoid-v4/training_env.pkl"

        # # Evaluation 2 -- Humanoid
        # net_path = "output/training_sac/2024-02-02/11-29-51Humanoid-v4/best_model.zip"
        # env_path = "output/training_sac/2024-02-02/11-29-51Humanoid-v4/training_env.pkl"

        # Evaluation 3 -- Humanoid
        net_path = "output/training_sac/2024-02-02/11-30-25Humanoid-v4/best_model.zip"
        env_path = "output/training_sac/2024-02-02/11-30-25Humanoid-v4/training_env.pkl"


        envs = make_parallel_envs(1)
        envs = VecNormalize.load(env_path,envs)

        eval_env = gymnasium.make(ENV_NAME, render_mode="human")
        eval_model = SAC.load(net_path, env=envs)

        num_episodes=5
        # mean_reward, std_reward = evaluate_policy(eval_model.policy, eval_env, n_eval_episodes=num_episodes, deterministic=True, render=True)
        # print('mean reward', mean_reward)
        # print('std', std_reward)

        for i in range(num_episodes):
            step = 0
            obs, _ = eval_env.reset()
            done = False
            while not done:
                actions, _states = eval_model.predict(envs.normalize_obs(obs), deterministic=True)
                obs, reward, trunc, term, info = eval_env.step(actions)
                
                if term or trunc:
                    done = True
                step += 1
            print("step length", step)

        # obs, _ = env.reset()
        # term = False
        # ep_len = 0
        # while term is False:
        #     action, _states = model.predict(env.unwrapped.normalize_obs(obs), deterministic=True)
        #     obs, rewards, term, trunc, info = env.step(action)
        #     ep_len += 1
        #     print('z coordinate of torso', obs[0])
        #     env.render()
        # print('Episode Length', ep_len)