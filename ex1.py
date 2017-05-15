
import numpy as np
import matplotlib.pyplot as plt

#          |20|
# |17|18|19| 0| 1| 2| 3|
# |16|     |21|     | 4|
# |15|              | 5|
# |14|     |22|     | 6|
# |13|12|11|10| 9| 8| 7|

states_given_up = {0:20,4:3,5:4,6:5,7:6,10:22,13:14,14:15,15:16,16:17,21:0}
states_given_down = {20:0,0:21,3:4,4:5,5:6,6:7,17:16,16:15,15:14,14:13}
states_given_left = {3:2,2:1,1:0,0:19,19:18,18:17,7:8,8:9,9:10,10:11,12:13}
states_given_right = {17:18,18:19,19:0,0:1,1:2,2:3,13:12,12:11,11:10,10:9,9:8,8:7}

actions_to_moves = {0:states_given_up,
                    1:states_given_down,
                    2:states_given_left,
                    3:states_given_right}

def init_P_and_R():
    P = np.zeros((4,23,23))
    for action in range(4):
        for state in range(23):
            move = actions_to_moves[action]
            if state in move:
                P[action,state,state] = 0.1
                if move[state] in move:
                    P[action,state,move[state]] = 0.75
                    P[action,state,move[move[state]]] = 0.15
                else:
                    P[action,state,move[state]] = 0.9
            else:
                P[action,state,state] = 1.
    R = np.zeros((4,23,23))
    R[1,0,21] = -100
    R[0,10,22] = 10
    return P,R

# Q1
# value iterations:

P,R = init_P_and_R()
V = np.zeros(23)
gamma = 0.95
maximal_time = 100

for t in range(maximal_time):
    new_V = np.max(np.sum(P*(R + gamma*np.tile(V,(4,23,1))),2),0)
    V = new_V

Q_star = np.sum(P*(R + gamma*np.tile(V,(4,23,1))),2)
#print(Q_star)
policy = np.argmax(Q_star,0)
print(policy)

# Q2
# convergency:

V = np.zeros(23)
epsilons = []

for t in range(maximal_time):
    Q_k = np.sum(P*(R + gamma*np.tile(V,(4,23,1))),2)
    new_V = np.max(Q_k,0)
    epsilons.append(np.mean(np.abs(Q_star-Q_k)))
    V = new_V

plt.plot(epsilons)
plt.show()

# Q3
# Q-learning

Q = np.random.rand(4,23)
state = 20
gamma = 0.95
alpha = 0.05
epsilon = 0.2
maximal_time = 100000
epsilons = []

def epsilon_greedy(values):
    action = np.argmax(values)
    if np.random.rand()<epsilon:
        action = np.random.choice(4)
    return action

for t in range(maximal_time):
    action = epsilon_greedy(Q[:,state])
    new_state = np.random.choice(23, p=(P[action,state,:]))
    reward = R[action,state,new_state]
    done = 0
    if new_state==22:
        new_state = 20
        done = 1
    Q[action,state] += alpha*(reward + (1-done)*gamma*np.max(Q[:,new_state]) - Q[action,state])
    epsilons.append(np.mean(np.abs(Q_star-Q)))
    state = new_state

#print(Q)
policy = np.argmax(Q,0)
print(policy)
plt.plot(epsilons)
plt.show()

# Q4
# SARSA

Q = np.random.rand(4,23)
state = 20
action = epsilon_greedy(Q[:,state])
gamma = 0.95
alpha = 0.05
epsilon = 0.2
maximal_time = 100000
epsilons = []

for t in range(maximal_time):

    new_state = np.random.choice(23, p=(P[action,state,:]))
    reward = R[action,state,new_state]
    done = 0
    if new_state==22:
        new_state = 20
        done = 1
    new_action = epsilon_greedy(Q[:,new_state])
    Q[action,state] += alpha*(reward + (1-done)*gamma*Q[new_action,new_state] - Q[action,state])
    epsilons.append(np.mean(np.abs(Q_star-Q)))
    state = new_state
    action = new_action

#print(Q)
policy = np.argmax(Q,0)
print(policy)
plt.plot(epsilons)
plt.show()
