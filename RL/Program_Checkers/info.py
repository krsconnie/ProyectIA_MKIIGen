import gym
import retro

env = retro.make(game='MortalKombatII-Genesis', players=1)
obs,_ = env.reset()

print("Observación:", type(obs), obs.shape)
action = env.action_space.sample()
obs, reward, done, _, info = env.step(action)
print("Recompensa:", reward)
print("Info extra:", info)


print(gym.__version__)
