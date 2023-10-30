from envs.box_pushing import BoxPushing


env = BoxPushing()

print(env.action_space)
print(env.observation_space)
print(env.observation_space.n)

print(env.reset())
print(env.step(5))