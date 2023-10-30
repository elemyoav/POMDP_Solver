from envs.box_pushing import BoxPushing


env = BoxPushing()

actions = [[i,j] for i in range(15) for j in range(15)]

actions2action = {
    tuple(actions): k for k, actions in enumerate(actions)
}



# 0 - N
# 1-  L
# 2 - R
# 3 - U
# 4 - D
# 5 - Sense small
# 6 - Sense large
# 7 - push small L
# 8 - push small R
# 9 - push small U
# 10 - push small D
# 11 - push large L
# 12 - push large R
# 13 - push large U
# 14 - push large D

print(env.reset())

while True:
    action = int(input())
    print(env.step(actions2action[(action,action)]))

