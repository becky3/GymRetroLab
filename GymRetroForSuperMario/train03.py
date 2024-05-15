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


class EpisodeCounterCallback(BaseCallback):
    def __init__(self):
        super(EpisodeCounterCallback, self).__init__()
        self.episode_count = 0
        self.interrupted = False

    def _on_step(self) -> bool:
        if 'done' in self.locals and self.locals['done']:
            self.episode_count += 1
            print(f"Episode: {self.episode_count}")

        if keyboard.is_pressed('ctrl+c'):
            print("Stopping training because 'Ctrl+C' key was pressed.")
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
        tensorboard_log: str,
        skip_learning: bool = False,
        policy: str = "MlpPolicy",
        n_steps: int = 2048,
) -> (bool, PPO):
    env = retro.make(game=game_name)
    env_train = DummyVecEnv([lambda: env])

    clip_range_value = linear_schedule(clip_rage_base)
    learning_rate_value = linear_schedule(learning_rate_base)

    if os.path.isfile(model_path):
        # Load the existing model
        model = PPO.load(
            model_path,
            policy=policy,
            env=env_train,
            verbose=1,
            learning_rate=learning_rate_value,
            ent_coef=ent_coef,
            clip_range=clip_range_value,
            device=device,
            n_steps=n_steps,
            tensorboard_log=tensorboard_log
        )
    else:
        # Create a new model
        model = PPO(
            policy=policy,
            env=env_train,
            verbose=1,
            learning_rate=learning_rate_value,
            ent_coef=ent_coef,
            clip_range=clip_range_value,
            device=device,
            n_steps=n_steps,
            tensorboard_log=tensorboard_log
        )

    episode_counter_callback = EpisodeCounterCallback()

    if not skip_learning:
        print("Start learning...")

        start_time = time.time()  # 学習開始時間を記録

        model.learn(
            total_timesteps=total_steps,
            callback=episode_counter_callback)

        print("Learning finished.")

        elapsed_time = time.time() - start_time
        print(f"Learning time: {round(elapsed_time)} seconds")

    model.save(model_path)

    env.close()

    return (not episode_counter_callback.interrupted,
            model)


def get_monitoring_env(
        env: retro.RetroEnv
) -> Monitor:

    now_str = datetime.now().strftime('%Y-%m-%d_%H%M%S%f')
    result_dir = os.path.join("./results", now_str)

    return Monitor(env, result_dir)


def play_game_and_save_video(
        model: PPO,
        game_name: str,
        is_save_video: bool = False
):
    env_play = retro.make(game=game_name)

    if is_save_video:
        env_play = get_monitoring_env(env_play)

    obs = env_play.reset()
    done = False

    while not done:
        if keyboard.is_pressed('ctrl+c'):
            break

        action, _states = model.predict(obs)
        obs, reward, done, info = env_play.step(action)
        env_play.render()

    env_play.render(close=True)
    env_play.close()


def main():

    is_skip_learning = True

    game_name = 'SuperMarioBros-Nes'
    model_path = './model/model02.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    model = None

    for i in range(100):
        print(f"Loop: {i+1}")

        training_successful, model = get_trained_model(
            skip_learning=is_skip_learning,
            game_name=game_name,
            total_steps=100000,
            clip_rage_base=0.3,
            learning_rate_base=0.01,
            ent_coef=1.0,
            model_path=model_path,
            device=device,
            policy="MlpPolicy",
            n_steps=2048*6,
            tensorboard_log="./ppo_tensorboard/"
        )

        if not training_successful or is_skip_learning:
            print("Training was stopped.")
            break

    play_game_and_save_video(
        is_save_video=True,
        model=model,
        game_name=game_name
    )


if __name__ == "__main__":
    main()
