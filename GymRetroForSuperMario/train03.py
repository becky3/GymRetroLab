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

    def _on_step(self) -> bool:
        if 'done' in self.locals and self.locals['done']:
            self.episode_count += 1
            print(f"Episode: {self.episode_count}")
        return True


def get_trained_model(
        game_name: str,
        total_steps: int,
        learning_rate: float,
        model_path: str,
        device: torch.device,
        skip_learning: bool = False,
        policy: str = "MlpPolicy",
        n_steps: int = 2048,
) -> PPO:
    env = retro.make(game=game_name)
    env_train = DummyVecEnv([lambda: env])

    if os.path.isfile(model_path):
        # Load the existing model
        model = PPO.load(
            model_path,
            policy=policy,
            env=env_train,
            verbose=2,
            learning_rate=learning_rate,
            device=device,
            n_steps=n_steps
        )
    else:
        # Create a new model
        model = PPO(
            policy=policy,
            env=env_train,
            verbose=2,
            learning_rate=learning_rate,
            device=device,
            n_steps=n_steps
        )

    if not skip_learning:
        episode_counter_callback = EpisodeCounterCallback()

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

    return model


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
        if keyboard.is_pressed('q'):
            break

        action, _states = model.predict(obs)
        obs, reward, done, info = env_play.step(action)
        env_play.render()

    env_play.render(close=True)
    env_play.close()


def main():

    game_name = 'SuperMarioBros-Nes'
    episode = 1000
    total_steps = int(100000 / 100 * episode)
    model_path = './model.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    model = get_trained_model(
        game_name=game_name,
        total_steps=total_steps,
        learning_rate=0.0025,
        model_path=model_path,
        skip_learning=True,
        device=device,
        n_steps=2048*4
    )
    play_game_and_save_video(
        model,
        game_name,
        is_save_video=True
    )


if __name__ == "__main__":
    main()
