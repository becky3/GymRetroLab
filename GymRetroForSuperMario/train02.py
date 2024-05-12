import retro
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    env = retro.make(game='SuperMarioBros-Nes')
    state_size = np.prod(env.observation_space.shape)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 1
    episodes = 1000
    for e in range(episodes):
        print("Episode: ", e)
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            # print("Start of loop iteration", time)
            env.render()
            # print("Before agent acts")
            action_index = agent.act(state)
            # print("After agent acts")
            action = np.zeros(env.action_space.n)
            action[action_index] = 1
            # print("Before environment step")
            next_state, reward, done, _ = env.step(action)
            # print("After environment step")
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            # print("Before agent remembers")
            agent.remember(state, action_index, reward, next_state, done)
            # print("After agent remembers")
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                    .format(e, episodes, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                # print("Before agent replays")
                agent.replay(batch_size)
                # print("After agent replays")
        if e % 10 == 0:
            agent.model.save_weights('dqn_weights.h5')

if __name__ == "__main__":
    main()