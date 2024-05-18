import gym
import retro
import keyboard
import time
from typing import Callable
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import Monitor
from datetime import datetime
import torch
import os
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env, rest_step: int):
        super(CustomRewardWrapper, self).__init__(env)
        self.pre_xscrollLo = 0
        self.pre_lives = 2
        self.pre_reward = 0
        self.total_reward = 0
        self.total_step = 0
        self.reset_step = rest_step
        self.env = env
        self.env.reset()

    def param_reset(self):
        self.pre_xscrollLo = 0
        self.pre_lives = 2
        self.pre_reward = 0
        self.total_reward = 0
        self.total_step = 0
        self.env.reset()

    def __del__(self):
        print("CustomRewardWrapper.__del__")
        self.env.render(close=True)
        self.env.close()

    def step(self, action: [int]):

        self.total_step += 1

        # 0: B
        # 1: NULL
        # 2: SELECT
        # 3: START
        # 4: UP
        # 5: DOWN
        # 6: LEFT
        # 7: RIGHT
        # 8: A

        for i in [1, 2, 3, 4, 5]:
            action[i] = 0

        # print(f"Action: {action}")

        state, reward, done, info = super().step(action)

        # if action[7] == 1:
        #     reward += 0.5

        lives = info["lives"]
        xscroll_lo = info["xscrollLo"]

        # reward -= 0.1
        scroll_reward = xscroll_lo - self.pre_xscrollLo
        need_reset = False

        if scroll_reward > 0:
            reward += scroll_reward
        elif scroll_reward == 0:
            reward -= 0.1
        if xscroll_lo == 0:
            reward -= 0.1

        if lives < self.pre_lives:
            reward -= 100
            self.pre_lives = lives
            need_reset = True

        if reward != self.pre_reward:
            print(
                f"reward:{reward} "
                + ", ".join([f"{key}: {value}" for key, value in info.items()])
            )

        self.pre_reward = reward
        self.pre_xscrollLo = xscroll_lo

        self.env.render()

        if 0 < self.reset_step <= self.total_step:
            need_reset = True

        self.total_reward += reward

        if need_reset:
            print("[Reset]total_reward: ", self.total_reward)
            self.param_reset()

        if done:
            print(f"Total reward for this episode was: {self.total_reward}")

        return state, reward, done, info


class EpisodeCounterCallback(BaseCallback):
    def __init__(self):
        super(EpisodeCounterCallback, self).__init__()
        self.episode_count = 0
        self.interrupted = False

    def _on_step(self) -> bool:

        if "done" in self.locals and self.locals["done"]:
            self.episode_count += 1
            return False

        if keyboard.is_pressed("ctrl+q"):
            print("Stopping training because 'Ctrl+Q' key was pressed.")
            self.interrupted = True
            return False

        return True


def linear_schedule(initial_value):
    def func(progress_remaining):
        return initial_value * progress_remaining

    return func


def get_trained_model(
    game_name: str,
    total_steps: int,
    clip_rage_base: float,
    learning_rate_base: float,
    ent_coef: float,
    model_path: str,
    device: torch.device,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    gamma: float,
    gae_lambda: float,
    tensorboard_log: str,
    policy: str,
) -> PPO:

    def make_env():
        env = retro.make(game=game_name)
        env = CustomRewardWrapper(env, rest_step=n_steps)
        return env

    env_maker: Callable[[], gym.Env] = make_env
    vec_env = SubprocVecEnv([env_maker] * 1)
    env_train = VecTransposeImage(VecFrameStack(vec_env, n_stack=4))

    clip_range_value = linear_schedule(clip_rage_base)
    learning_rate_value = linear_schedule(learning_rate_base)

    ppo_params = {
        "policy": policy,
        "env": env_train,
        "verbose": 2,
        "learning_rate": learning_rate_value,
        "ent_coef": ent_coef,
        "clip_range": clip_range_value,
        "device": device,
        "n_steps": n_steps,
        "tensorboard_log": None if tensorboard_log == "" else tensorboard_log,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
    }

    if os.path.isfile(model_path):
        # Load the existing model
        model = PPO.load(model_path, **ppo_params)
    else:
        # Create a new model
        model = PPO(**ppo_params)

    print("Start learning...")

    start_time = time.time()

    episode_counter_callback = EpisodeCounterCallback()

    model.learn(total_timesteps=total_steps, callback=episode_counter_callback)

    print("Learning finished.")

    elapsed_time = time.time() - start_time
    print(f"Learning time: {round(elapsed_time)} seconds")

    start_time = time.time()

    model.save(model_path)

    vec_env.close()

    elapsed_time = time.time() - start_time
    print(f"Model saving time: {round(elapsed_time)} seconds")

    return model


def get_monitoring_env(env: retro.RetroEnv, index: int) -> Monitor:

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S%f")
    result_dir = os.path.join("./results", now_str) + "_" + str(index).zfill(3)

    return Monitor(env, result_dir)


def play_game_and_save_video(
    model: PPO, game_name: str, index: int, is_save_video: bool = False
):
    def make_env():
        env = retro.make(game=game_name)
        env = CustomRewardWrapper(env)
        if is_save_video:
            env = get_monitoring_env(env, index)
        return env

    env_maker: Callable[[], gym.Env] = make_env
    vec_env = SubprocVecEnv([env_maker] * 1)
    env_play = VecTransposeImage(VecFrameStack(vec_env, n_stack=4))

    obs = env_play.reset()
    done = False

    while not done:
        if keyboard.is_pressed("ctrl+q"):
            break

        action, _states = model.predict(obs)
        obs, reward, done, info = env_play.step(action)
        # env_play.render()

    vec_env.close()


def main():

    game_name = "SuperMarioBros-Nes"
    model_path = "./model/model.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    step = 20000

    for i in range(50):
        index = i + 1
        print(f"Loop: {index}")

        model = get_trained_model(
            game_name=game_name,
            clip_rage_base=0.05,
            learning_rate_base=0.05,
            ent_coef=0.3,
            model_path=model_path,
            device=device,
            policy="CnnPolicy",
            n_steps=3000,
            total_steps=step,
            batch_size=100,
            n_epochs=20,
            gamma=0.95,
            gae_lambda=0.90,
            tensorboard_log="",  # "./ppo_tensorboard/",
        )

        if index % 50 == 0:
            play_game_and_save_video(
                is_save_video=True, model=model, index=index, game_name=game_name
            )


if __name__ == "__main__":
    main()
