import retro
import keyboard
from stable_baselines3 import PPO
from gym.wrappers import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from datetime import datetime
import torch
import os



def get_trained_model(
        game_name: str,
        total_steps: int,
        learning_rate: float,
        model_path: str,
        device: torch.device,
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

    model.learn(total_timesteps=total_steps)

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
    total_steps = 100000
    model_path = './model.pkl'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    model = get_trained_model(
        game_name=game_name,
        total_steps=total_steps,
        learning_rate=0.005,
        model_path=model_path,
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
