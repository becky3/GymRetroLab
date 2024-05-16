import gym
import retro
import keyboard
import time
from stable_baselines3 import PPO
from gym.wrappers import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
import torch
import os


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super(CustomRewardWrapper, self).__init__(env)
        self.pre_xscrollLo = 0
        self.pre_lives = 2
        self.pre_reward = 0
        self.total_reward = 0

    def step(self, action):
        state, reward, done, info = super().step(action)

        lives = info["lives"]
        xscroll_lo = info["xscrollLo"]

        reward -= 1
        scroll_reward = xscroll_lo - self.pre_xscrollLo

        if scroll_reward > 0:
            reward += scroll_reward * 20000
        elif scroll_reward == 0:
            reward -= 20

        if lives < self.pre_lives:
            reward -= 200000
            self.pre_lives = lives
            if xscroll_lo == 0:
                reward -= 20000000
            done = True

        if reward != self.pre_reward:
            print(
                f"reward:{reward} "
                + ", ".join([f"{key}: {value}" for key, value in info.items()])
            )
            self.pre_reward = reward

        self.pre_xscrollLo = xscroll_lo
        self.total_reward += reward

        # エピソードが終了したときに累積報酬を出力し、リセットする
        if done:
            print(f"Total reward for this episode was: {self.total_reward}")
            self.total_reward = 0

        return state, reward, done, info


class EpisodeCounterCallback(BaseCallback):
    def __init__(self, env):
        super(EpisodeCounterCallback, self).__init__()
        self.episode_count = 0
        self.interrupted = False
        self.env = env  # Add a reference to the environment

    def _on_step(self) -> bool:
        self.env.render()  # Render the environment at each step

        if "done" in self.locals and self.locals["done"]:
            self.episode_count += 1
            return False

        if keyboard.is_pressed("ctrl+q"):
            print("Stopping training because 'Ctrl+Q' key was pressed.")
            self.interrupted = True
            return False  # Return False to stop training

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
    tensorboard_log: str = None,
    skip_learning: bool = False,
    policy: str = "MlpPolicy",
    n_steps: int = 2048,
) -> (bool, PPO):
    env = retro.make(game=game_name)
    env = CustomRewardWrapper(env)
    env_train = DummyVecEnv([lambda: env])

    clip_range_value = linear_schedule(clip_rage_base)
    learning_rate_value = linear_schedule(learning_rate_base)

    if os.path.isfile(model_path):
        # Load the existing model
        model = PPO.load(
            model_path,
            policy=policy,
            env=env_train,
            verbose=2,
            learning_rate=learning_rate_value,
            ent_coef=ent_coef,
            clip_range=clip_range_value,
            device=device,
            n_steps=n_steps,
            tensorboard_log=tensorboard_log,
        )
    else:
        # Create a new model
        model = PPO(
            policy=policy,
            env=env_train,
            verbose=2,
            learning_rate=learning_rate_value,
            ent_coef=ent_coef,
            clip_range=clip_range_value,
            device=device,
            n_steps=n_steps,
            tensorboard_log=tensorboard_log,
        )

    episode_counter_callback = EpisodeCounterCallback(env)

    if not skip_learning:
        print("Start learning...")

        start_time = time.time()  # 学習開始時間を記録

        model.learn(total_timesteps=total_steps, callback=episode_counter_callback)

        print("Learning finished.")

        elapsed_time = time.time() - start_time
        print(f"Learning time: {round(elapsed_time)} seconds")
    else:
        print("Skip learning.")

    model.save(model_path)

    env.render(close=True)
    env.close()

    return not episode_counter_callback.interrupted, model


def get_monitoring_env(env: retro.RetroEnv, index: int) -> Monitor:

    now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S%f")
    result_dir = os.path.join("./results", now_str) + "_" + str(index).zfill(3)

    return Monitor(env, result_dir)


def play_game_and_save_video(
    model: PPO, game_name: str, index: int, is_save_video: bool = False
):
    env_play = retro.make(game=game_name)

    if is_save_video:
        env_play = get_monitoring_env(env_play, index)

    env_play = CustomRewardWrapper(env_play)

    obs = env_play.reset()
    done = False

    while not done:
        if keyboard.is_pressed("ctrl+q"):
            break

        action, _states = model.predict(obs)
        obs, reward, done, info = env_play.step(action)
        env_play.render()

    env_play.render(close=True)
    env_play.close()


def main():

    is_skip_learning = False

    game_name = "SuperMarioBros-Nes"
    model_path = "./model/model.pkl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    for i in range(500):
        index = i + 1
        print(f"Loop: {index}")

        training_successful, model = get_trained_model(
            skip_learning=is_skip_learning,
            game_name=game_name,
            total_steps=20000,
            clip_rage_base=0.3,
            learning_rate_base=0.3,
            ent_coef=0.5,
            model_path=model_path,
            device=device,
            policy="MlpPolicy",
            n_steps=2048 * 10,
            # tensorboard_log="./ppo_tensorboard/",
        )

        if index == 1 or index % 50 == 0:
            play_game_and_save_video(
                is_save_video=True, model=model, index=index, game_name=game_name
            )

        if not training_successful or is_skip_learning:
            print("Training was stopped.")
            break


if __name__ == "__main__":
    main()
