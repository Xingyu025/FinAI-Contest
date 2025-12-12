# Standardize the environment

## Action Space
The action space includes three discrete actions:
- **0 — Hold**
- **1 — Buy**
- **2 — Sell**

Internally, the environment sets:
'''python
self.action_space = np.array([3,])
self.gym_actions = range(3)
'''

Action constraints:
The agent **cannot exceed** the maximum allowed position ‘max_position’.
If attempting to buy while already at ‘+max_position’, action becomes **hold**.
If attemping to sell while already at ‘-max_postion’, action becomes **hold**.

## State Space

## Observation Space

## Reward Design

## Transition Dynamics

## Initial Condition
