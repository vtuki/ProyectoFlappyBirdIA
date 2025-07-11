from entornoFlappy import FlappyEnv
from agenteQLearning import QAgent

env = FlappyEnv()
agent = QAgent()

episodes = 1000

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for t in range(1000):
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)

        state = next_state
        total_reward += reward

        if done:
            break

    print(f'Episode {episode}: Score {total_reward}')