import retro
import keyboard
from stable_baselines3 import PPO
from gym.wrappers import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    # 環境の作成
    env = retro.make(game='SuperMarioBros-Nes')

    # Monitorでラップし、結果を保存するディレクトリを指定
    # force=Trueを追加して、以前のモニターファイルを自動的にクリア
    env = Monitor(env, "./results", force=True)

    env_train = DummyVecEnv([lambda: env])

    print("starting game...")

    # PPOエージェントの作成
    model = PPO("MlpPolicy", env_train, verbose=1)

    # エージェントの訓練
    model.learn(total_timesteps=100000)

    print("Training finished.")

    # 訓練用の環境を閉じる
    env.close()

    # 訓練したエージェントでプレイ
    env_play = retro.make(game='SuperMarioBros-Nes')  # プレイ用の環境
    obs = env_play.reset()
    done = False
    while not done:
        # キーボードで 'q' が押された場合はループを抜ける
        if keyboard.is_pressed('q'):
            print("Quit key pressed. Exiting game...")
            break
        
        action, _states = model.predict(obs)
        obs, reward, done, info = env_play.step(action)
        env_play.render()

    print("Game finished.")

    # Monitorを閉じる前に、内部の環境を閉じます
    env_play.render(close=True)
    env_play.close()

    print("Environment closed.")

if __name__ == "__main__":
    main()