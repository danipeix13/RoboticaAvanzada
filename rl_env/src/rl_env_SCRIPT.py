from EnvKinova import *


if __name__ == "__main__":
    EPOCHS = 100
    STEPS_PER_EPOCH = 1000

    rl_env = EnvKinova() # DONE
    for i in range(EPOCHS):
        observation = rl_env.reset()
        for ts in range(STEPS_PER_EPOCH):
            print(observation)
            action = rl_env.action_space_sample()
            observation, reward, done, info = rl_env.step(action)
            if done:
                print(f"Episode finished after {ts+1} timestamps")
                break
    rl_env.close() # DONE
