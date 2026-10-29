# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement
Implement the Value Iteration algorithm on the FrozenLake environment to determine the optimal value of each state and derive the optimal policy that maximizes the expected cumulative reward. The algorithm should repeatedly update state values until convergence and then extract the best action for every state.




## Software Requirements
```
Python 
Gymnasium Library
NumPy
Jupyter Notebook/Google colab
```
## Environment Description

The environment is a 4×4 customized FrozenLake consisting of safe frozen cells (F), hole states (H), a start state (S), and a goal state (G). The agent begins at the Start (S) and must reach the Goal (G) while avoiding the holes. Since the environment uses is_slippery=True, the agent may not always move in the intended direction, making the environment stochastic. The Value Iteration algorithm is used to compute the optimal state-value function and the optimal policy for safely reaching the goal.
### Map Used:
```

F F H F
H S H F
H F F H
H F G H

```
```
Where:

S – Start State
F – Frozen Surface (Safe)
H – Hole (Terminal State)
G – Goal State (Reward = 1)

Since is_slippery=True, the agent may move in unintended directions due to slippery ice, making the environment stochastic. 
```
## MDP Representation

#### States (S)
There are 16 states in a 4×4 FrozenLake grid.

S={0,1,2,…,15}
Each grid cell represents one state. The agent starts from the Start (S) state and aims to reach the Goal (G) state while avoiding Hole (H) states.

#### Actions (A)

There are four possible actions.
```
Action	Value
Left	0
Down	1
Right	2
Up	3
```
### Transition Probability (P)
The transition probability defines the likelihood of moving from one state to another after taking an action.
Since is_slippery=True, the environment is stochastic, meaning the agent may move in an unintended direction. Therefore, the next state depends on both the selected action and the slipping probability.
``` 
P(s′∣s,a)
```

### Reward Function (R)
```
Goal State = +1
Hole = 0
Frozen State = 0
Discount Factor (γ)
γ=0.99
```
The discount factor gives high importance to future rewards.

## Theory
Value Iteration is a dynamic programming algorithm used to solve Markov Decision Processes (MDPs) by repeatedly applying the Bellman Optimality Equation until the value function converges. In each iteration, the algorithm updates the value of every state by selecting the action that provides the maximum expected future reward. Once the values converge, the optimal policy is obtained by choosing the action with the highest value for each state.
The Bellman Optimality Equation is:

```
V(s) = max_a Σ P(s' | s, a) [R(s, a, s') + γV(s')]
```

## Algorithm

1. Create the customized FrozenLake-v1 environment using the modified map.
2. Initialize the value of all states to zero.
3. Set the discount factor (γ = 0.99) and convergence threshold (θ = 1e−8).
4. Compute the expected value for every possible action in each state using the Bellman Optimality Equation.
5. Update each state's value with the maximum expected action value.
6. Repeat the updates until the change in state values is less than the convergence threshold.
7. Extract the optimal policy by selecting the action with the highest expected value for each state.
8. Display the number of iterations, optimal state-value function, and optimal policy.



## Python Program

```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [
    "FFHF",
    "HSHF",
    "HFFH",
    "HFGH"
]

env = gym.make("FrozenLake-v1", desc=env_desc, is_slippery=True)
def value_iteration(env, gamma=0.99, theta=1e-8):
    """
    Performs value iteration and returns the optimal value function.
    """
    nS = env.observation_space.n
    nA = env.action_space.n

    V = np.zeros(nS)
    iteration = 0

    while True:
        delta = 0

        for s in range(nS):
            action_values = np.zeros(nA)

            for a in range(nA):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            best_action_value = np.max(action_values)
            delta = max(delta, abs(best_action_value - V[s]))
            V[s] = best_action_value

        iteration += 1

        if delta < theta:
            break

    # Extract Optimal Policy
    policy = np.zeros(nS, dtype=int)

    for s in range(nS):
        action_values = np.zeros(nA)

        for a in range(nA):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                action_values[a] += prob * (
                    reward + gamma * V[next_state] * (not done)
                )

        policy[s] = np.argmax(action_values)

    return V, policy, iteration
# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

# write your code here


V,policy,iterations=value_iteration(env)





# -------------------------------------------------
# Display Output
# -------------------------------------------------
print("Name: JAIHARI S")
print("Register Number: 212225230109")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))


action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

env.close()








```

---

## Output


### Number of Iterations:

#### For Normal description:

<img width="357" height="115" alt="image" src="https://github.com/user-attachments/assets/0acd3603-e15f-49e7-ad4e-e73a5a80da9c" />





#### After changing the description:

<img width="377" height="116" alt="image" src="https://github.com/user-attachments/assets/ec60fc28-66fa-4ec0-a14d-c7c7fef946cb" />









### Optimal State-Value Function:

#### For Normal description:

<img width="226" height="90" alt="image" src="https://github.com/user-attachments/assets/a58d9ed3-220f-42cf-8a2e-10e0f7e754f1" />

#### After changing the description: 
<img width="223" height="87" alt="image" src="https://github.com/user-attachments/assets/feab754a-8ee0-46be-89c9-4e2afbcbeded" />



### Optimal Policy:

#### For Normal description:
<img width="181" height="77" alt="image" src="https://github.com/user-attachments/assets/b35de0b7-bd1f-4e1d-82c4-8f7c61a7cbf2" />


#### After changing description:
<img width="175" height="95" alt="image" src="https://github.com/user-attachments/assets/1e0ed09e-e87d-4f99-83db-25340376a91e" />




## Result

The Value Iteration algorithm was successfully implemented on the customized FrozenLake-v1 environment. It converged after several iterations and computed the optimal state-value function and optimal policy for the modified map. The resulting policy guides the agent from the start state to the goal while avoiding hole states as much as possible.

## Inference
The comparison between the original and modified FrozenLake environments shows that changing the map layout directly affects the Markov Decision Process. In the modified environment, the start state, goal state, and hole positions were rearranged, creating different state transitions and fewer safe paths. As a result, the optimal state-value function, policy, and convergence behavior differed from the original environment. The Value Iteration algorithm successfully adapted to these changes by recalculating the optimal values and selecting the best action for each state. This confirms that Value Iteration is robust and capable of finding the optimal policy for different MDP configurations.
