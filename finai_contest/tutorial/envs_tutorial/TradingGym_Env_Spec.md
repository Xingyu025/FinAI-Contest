# Standardize the environment

## Action Space
The action space includes three discrete actions:
- **0 — Hold**
- **1 — Buy**
- **2 — Sell**

Internally, the environment sets:
```python
self.action_space = np.array([3,])
self.gym_actions = range(3)
```

Action constraints:
The agent **cannot exceed** the maximum allowed position `max_position`.
If attempting to buy while already at `+max_position`, action becomes **hold**.
If attemping to sell while already at `-max_postion`, action becomes **hold**.

## State Space
### **Market Data**
- **Market data segment**(`df_sample`): defines the time range over which the episode is executed.
- **Stock price**(`price`): is used as the reference for all reward and P&L calculations.
- **Market feature matrix**(`obs_features`): A two-dimensional array containing the selected market features at each time step.
### **Position State**
- **Market position**(`posi_arr`): A one-dimensional array representing the agent's position at each time step, measured in number of shares of contracts.
- **Position variation**(`posi_variating_arr`): records how the position changes due to trading actions, such as increasing or decreasing exposure.
- **Position entry/ cover indicator**(`posi_entry_cover_arr`): A one-dimensional array encoding whether a position change corresponds to an entry or a cover operation.
### **Price Statistics**
- **Average position price**(`price_mean_arr`): A one-dimensional array representing the average entry price of current position at each time step.
### **Reward and Profit & Loss**
- **Unrealized reward**(`reward_fluctuant_arr`): defined as the difference between the current price and the average position price, multiplied by the current position size.
- **Realized reward indicator**(`reward_makereal_arr`): A one-dimensional arry indicating whether the reward at a given time step is realized.
- **Realized reward**(`reward_arr`): A one-dimensional array representing the realized profit or loss at each time step.
## **Control and Bookkeeping**
- **Step pointer**(`step_st`): An integer indicating the starting index of the current observation window within the trading session.
- **Additional information**(`info`): A placeholder variable for returning auxiliary information when the episode terminates.
- **Transaction log**(`transaction_details`): An empty DataFrame used to record detailed information about all trading action, positions, and rewards during the episode.

## Observation Space

## Reward Design

## Transition Dynamics

## Initial Condition
