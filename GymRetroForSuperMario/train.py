import retro
import keyboard

def main():
    # 環境の作成
    env = retro.make(game='SuperMarioBros-Nes')
    obs = env.reset()

    print("starting game...")

    done = False
    while not done:
        # キーボードで 'q' が押された場合はループを抜ける
        if keyboard.is_pressed('q'):
            print("Quit key pressed. Exiting game...")
            break
        
        # ここではランダムなアクションを選択
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        # ゲームの状態を画面に表示
        env.render()

    # 環境を閉じる

    print("Game finished.")

    env.render(close=True)
    env.close()

    print("Environment closed.")

if __name__ == "__main__":
    main()
