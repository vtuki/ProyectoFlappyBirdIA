from entornoFlappy import FlappyEnv
from agenteQLearning import QAgent
from entornoFlappy import FACTOR_DISCRETIZACION_X, FACTOR_DISCRETIZACION_Y

env = FlappyEnv()
agent = QAgent()

OVER_TRAIN = True

path = "models/qtable123Pipes-40ScoreBEST.pkl"
if OVER_TRAIN:
    agent.load(path)
    agent.epsilon = 0

episodes = 10000

max_surpassed_pipes = 0
mean_score = 0
for episode in range(episodes):
    state = env.reset()
    state_changed = True
    total_reward = 0
    surpased_pipes = 0
    done = False

    while not done:
        action = agent.choose_action(state)
        action_done = action
        if (not state_changed):
            action_done = 0
        else:
            state_changed = False

        next_state, reward, done, pasoTuberia = env.step(action_done)

        if (state != next_state or done):

            if pasoTuberia:
                surpased_pipes+=1

            agent.learn(state, action, reward, next_state)

            state_changed = True
            mean_score += reward
            state = next_state
            total_reward += reward
        
        if (not state_changed and pasoTuberia):
            surpased_pipes += 1
            agent.learn(state, action, reward, next_state)
            total_reward += reward
            mean_score+=reward
        
    
    if surpased_pipes > max_surpassed_pipes:
        max_surpassed_pipes = surpased_pipes

    agent.decay_epsilon()
    print(f'Episode {episode}: Score {total_reward}, Tuberias superadas: {surpased_pipes}, eps: {agent.epsilon}')


mean_score /= episodes
agent.save(f"models/qtable{max_surpassed_pipes}Pipes{round(mean_score)}Score.pkl")
if OVER_TRAIN:
    agent.saveMetaData(episodes, FACTOR_DISCRETIZACION_X, FACTOR_DISCRETIZACION_Y, f"models/metadata{max_surpassed_pipes}Pipes{round(mean_score)}Score.txt", True)
else:
    agent.saveMetaData(episodes, FACTOR_DISCRETIZACION_X, FACTOR_DISCRETIZACION_Y, f"models/metadata{max_surpassed_pipes}Pipes{round(mean_score)}Score.txt")
print(f"Máxima cantidad de tuberias superadas: {max_surpassed_pipes}, Score medio: {mean_score}")