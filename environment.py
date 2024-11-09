import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import deque
import random
import time

from vmas.simulator.core import Agent, Box, World
from vmas.simulator.core import Agent
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.dynamics.holonomic_with_rot import HolonomicWithRotation
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils
from vmas.simulator.rendering import Color, Line, Transform
from vmas.simulator import rendering
from vmas import make_env

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        self.max_action = max_action

    def forward(self, state):
        return torch.tanh(self.network(state)) * self.max_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Network for Q1 value
        self.network1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Network for Q2 value
        self.network2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        return self.network1(xu), self.network2(xu)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def perform_updates(batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, actors, target_actors, critics, target_critics, optimizer_actors, optimizer_critics, gamma):
    # Convert lists to tensors
    states = torch.FloatTensor(batch_states)
    next_states = torch.FloatTensor(batch_next_states)
    actions = torch.FloatTensor(batch_actions)
    rewards = torch.FloatTensor(batch_rewards).unsqueeze(1)
    dones = torch.FloatTensor(batch_dones).unsqueeze(1)

    # Process actions if necessary
    if actions.dim() == 3:  # Assuming [batch_size, num_agents, action_dim]
        actions = actions.mean(dim=1)  # Example: mean across agents

    # Compute the target Q values
    with torch.no_grad():
        next_actions = [target_actor(next_states) for target_actor in target_actors]
        Q_next_values = [torch.min(*target_critic(next_states, na)) for target_critic, na in zip(target_critics, next_actions)]
        Q_next = torch.cat(Q_next_values, dim=1).min(dim=1, keepdim=True)[0]
        Q_target = rewards + (1 - dones) * gamma * Q_next

    # Get the current Q-values from the critic
    Q_current_values = [torch.min(*critic(states, actions)) for critic in critics]
    Q_current = torch.cat(Q_current_values, dim=1).mean(dim=1, keepdim=True)

    # Calculate critic loss and update
    critic_loss = F.mse_loss(Q_current, Q_target)
    for optimizer in optimizer_critics:
        optimizer.zero_grad()
    critic_loss.backward()
    for optimizer in optimizer_critics:
        optimizer.step()

    # Update actors
    actor_losses = []
    for actor, optimizer in zip(actors, optimizer_actors):
        optimizer.zero_grad()
        predicted_actions = actor(states)
        actor_loss = -torch.mean(torch.stack([critic(states, predicted_actions)[0] for critic in critics]))
        actor_loss.backward()
        optimizer.step()
        actor_losses.append(actor_loss.item())

    return np.mean(actor_losses), critic_loss.item()

class GaussianNoise:
    def __init__(self, action_dimension, mu=0.5, sigma=0.05):
        self.action_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        return np.random.normal(self.mu, self.sigma, self.action_dimension)

state_dim = 15
action_dim = 10  # Ensure this is consistent with IoVEnvironment setup
max_action = 1

num_agents = 2
num_platoons = 2
cars_per_platoon = 2

# Initialization of Actors, Critics, and Optimizers
actors = []
critics = []
target_actors = []
target_critics = []
optimizer_actors = []
optimizer_critics = []

for _ in range(num_agents):
    actor = Actor(state_dim, action_dim, max_action)
    critic = Critic(state_dim, action_dim)
    target_actor = Actor(state_dim, action_dim, max_action)
    target_critic = Critic(state_dim, action_dim)

    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    optimizer_actor = optim.SGD(actor.parameters(), lr=1e-6, momentum=0.9)
    optimizer_critic = optim.SGD(critic.parameters(), lr=1e-6, momentum=0.8)


    actors.append(actor)
    critics.append(critic)
    target_actors.append(target_actor)
    target_critics.append(target_critic)
    optimizer_actors.append(optimizer_actor)
    optimizer_critics.append(optimizer_critic)

replay_buffer_capacity = 1_000_000  # Example capacity
replay_buffer = ReplayBuffer(replay_buffer_capacity)

class RSU:
    def __init__(self, position, coverage_radius):
        self.position = torch.tensor(position, dtype=torch.float32)  # Ensure position is a tensor
        self.coverage_radius = coverage_radius

    def is_within_coverage(self, position):
        # Convert position (tuple) to a torch tensor if it isn't already
        position_tensor = torch.tensor(position, dtype=torch.float32) if isinstance(position, tuple) else position
        
        # Calculate the distance and check if it's within coverage
        return torch.norm(self.position - position_tensor) <= self.coverage_radius

class CarAgent(Agent):
    def __init__(
        self,
        index,
        position,
        speed,
        direction_angle,
        name="",
        power= 0.1,
        channels=[],
        channel_gain=0.1,
        snr=0.1,
        rate=0,
        leader=None,
        distance=0,
        world=None,
        direction=None
    ):
        """
        Initializes the CarAgent with position, speed, communication parameters, and optional leader.

        Parameters:
        - index (int): Unique identifier for the agent.
        - position (list or np.array): Initial position of the car agent.
        - speed (float): Initial speed of the car agent.
        - name (str): Name identifier for the agent.
        - power (float): Power level allocated for communication.
        - channels (list): List of available communication channels.
        - channel_gain (float): Initial channel gain value.
        - snr (float): Initial Signal-to-Noise Ratio.
        - rate (float): Initial data transmission rate.
        - leader (CarAgent): The leader agent that this car follows, if any.
        - distance (float): Initial distance to the RSU or other reference point.
        """
        super().__init__(name=name)
        self.index = index
        self.position = np.array(position, dtype=float)
        self.speed = speed
        self.direction_angle = np.deg2rad(direction_angle)
        self.power = power
        self.snr = snr
        self.rate = rate
        self.channels = channels
        self.channel_gain = channel_gain
        self.assigned_channel = None
        self.noise_level = 1e-14 
        self.leader = leader
        self.distance = distance 
        self.max_rate = rate  
        self.init_spacing()
   
    def init_spacing(self):
        # Ensure these match the dimensions used for visual rendering
        self.lane_width = 1  # Same as road_width in extra_render
        self.square_size = 1.5  # Same as square_size in extra_render
        self.gap_size = 0.1     # Same as gap_size in extra_render
        self.center_x = 0
        self.center_y = 0

        # Recalculate boundaries based on updated sizes
        self.left_x = self.center_x - self.square_size
        self.right_x = self.center_x + self.square_size
        self.top_y = self.center_y + self.square_size
        self.bottom_y = self.center_y - self.square_size
        self.gap_left_x = self.center_x - self.gap_size / 2
        self.gap_right_x = self.center_x + self.gap_size / 2
        self.gap_top_y = self.center_y + self.gap_size / 2
        self.gap_bottom_y = self.center_y - self.gap_size / 2    
    
    def move(self):
        """
        Updates the agent's position based on its current speed and direction.

        The position is incremented by the value of the speed scaled by a small time delta.
        """
        delta_time = 0.1  # Smaller time steps for smoother movement
        position_array = np.array(self.position, dtype=float)
        speed_value = self.speed * delta_time  # Apply speed with time scaling for smoothness
        position_array[0] += speed_value  # Move along the x-axis
        self.position = tuple(position_array)


    def update(self):
        """
        Main update function called each simulation step. Manages position updates and interaction with the leader.
        """
        if self.leader:
            self.follow_leader()
        else:
            self.move()
        
        # Ensure the position is within lane boundaries after any adjustments
        if not self.is_within_lane(*self.position):
            self.position = self.adjust_position_within_bounds(*self.position)

    def follow_leader(self):
        """
        Modifies the agent's behavior to follow a leader while maintaining a safe following distance.
        """
        # Calculate the direction to the leader and the distance
        leader_pos = np.array(self.leader.position)
        my_pos = np.array(self.position)
        direction_to_leader = leader_pos - my_pos
        distance_to_leader = np.linalg.norm(direction_to_leader)
        
        # Move towards the leader if further away than the safe following distance
        if distance_to_leader > self.safe_distance:
            normalized_direction = direction_to_leader / distance_to_leader
            move_step = normalized_direction * self.speed
            projected_x, projected_y = my_pos + move_step
            # Validate the new position
            self.position = self.get_valid_position(projected_x, projected_y, 0, 0)
        else:
            # Adjust speed or maintain current position to match leader's speed
            self.speed = self.leader.speed

    def get_valid_position(self, current_x, current_y, delta_x, delta_y):
        """
        Calculate the next valid position ensuring it does not exceed road boundaries.
        """
        projected_x = current_x + delta_x
        projected_y = current_y + delta_y

        # Ensure the proposed position is within the road boundaries
        return self.adjust_position_within_bounds(projected_x, projected_y)

    def is_within_lane(self, x, y):
        """
        Checks if the given position (x, y) is within the allowed lane boundaries.
        """
        return self.left_x + self.lane_width <= x <= self.right_x - self.lane_width and \
               self.bottom_y + self.lane_width <= y <= self.top_y - self.lane_width

    def adjust_position_within_bounds(self, x, y):
        """
        Adjusts the position to ensure it remains within the defined road boundaries.
        """
        adjusted_x = np.clip(x, self.left_x, self.right_x)
        adjusted_y = np.clip(y, self.bottom_y, self.top_y)
        return adjusted_x, adjusted_y
       
    def stable_matching(self, agent, channels, preferences):
        channel_rank = {}
        for ch in channels:
            if ch.frequency in preferences:
                channel_rank[ch.frequency] = {ag: idx for idx, ag in enumerate(preferences[ch.frequency])}
            else:
                channel_rank[ch.frequency] = {agent: 0}  # default or error handling case

        proposals = {ag: deque(preferences[ag]) for ag in [agent]}
        pairs = {}
        free_agents = set([agent])
        
        while free_agents:
            ag = free_agents.pop()
            while proposals[ag]:
                ch = proposals[ag].popleft()
                if ch not in pairs:
                    pairs[ch] = ag
                    break
                else:
                    current_agent = pairs[ch]
                    if channel_rank[ch][ag] < channel_rank[ch][current_agent]:
                        free_agents.add(current_agent)
                        pairs[ch] = ag
                        break
                    else:
                        if not proposals[ag]:
                            free_agents.add(ag)

        return {ag: ch for ch, ag in pairs.items()}


    def greedy_subcarrier_allocation(self, agent, channels, data_rates):
        if channels: 
            best_channel = max(channels, key=lambda x: data_rates[(agent, x)])
            allocation = {agent: best_channel}
            channels.remove(best_channel)
        else:
            allocation = {agent: None}  # No channels available

        return allocation

    def worst_user_avoidance(self, agent, channels, channel_conditions):
        allocation = {}

        if channels:  
            best_channel = min(channels, key=lambda x: channel_conditions[(agent, x)])
            allocation[agent] = best_channel

        return allocation

    def assign_channel(self, method, agent, channels, data=None):
        if method == 'Stable':
            preferred_channel = next((c for c in channels if c.frequency == 2), None)
            if preferred_channel:
                other_channels = [c for c in channels if c != preferred_channel]
                preferences = {agent: [preferred_channel] + other_channels}
            else:
                preferences = {agent: channels}
            allocation = self.stable_matching(agent, channels, preferences)
        elif method == 'Greedy':
            allocation = self.greedy_subcarrier_allocation(agent, channels, data)
        elif method == 'WUA':
            allocation = self.worst_user_avoidance(agent, channels, data)
        else:
            raise ValueError("Invalid method. Must be one of 'stable', 'greedy', or 'wua'.")

        self.assigned_channel = allocation[agent]
    
    def proportional_power_algorithm(self, user_index, priorities, weights, gain_matrix, max_power):
        """
        Calculates the optimal power allocation based on priorities, weights, and gain matrix.

        Parameters:
        - user_index (int): Identifier for the current user (agent) to calculate power allocation for.
        - priorities (list): List of priority values for each agent.
        - weights (list): List indicating the load on each channel.
        - gain_matrix (dict): Dictionary mapping (user, channel) tuples to channel gains.
        - max_power (float): The maximum power that can be allocated.

        Returns:
        - optimal_power (float): The optimal power allocation for the agent.
        """
        # Validate user_index
        if user_index is None or not isinstance(user_index, int) or user_index < 0 or user_index >= len(priorities):
            return max_power

        s = weights[user_index]  # Current user's weight
        t = weights[user_index - 1] if user_index > 0 else None  # Previous user's weight

        if user_index == 0 or t is None or s >= t:
            return max_power

        # Assuming 's' and 't' are channel indices or identifiers
        # Ensure that 's' and 't' are valid channels
        current_channel = self.channels[s] if s < len(self.channels) else None
        previous_channel = self.channels[t] if t < len(self.channels) else None

        if current_channel and (user_index, current_channel.frequency) in gain_matrix:
            gain_uu_s = gain_matrix[(user_index, current_channel.frequency)]
        else:
            gain_uu_s = 0  # Default or handle appropriately

        if previous_channel and (user_index, previous_channel.frequency) in gain_matrix:
            gain_uu_t = gain_matrix[(user_index, previous_channel.frequency)]
        else:
            gain_uu_t = 0  # Default or handle appropriately

        # Prevent division by zero
        if (s - t) == 0:
            optimal_power = max_power
        else:
            optimal_power = (t * gain_uu_s - s * gain_uu_t) / (s - t)
            optimal_power = max(0, min(optimal_power, max_power))

        return optimal_power


    def assign_power(self, method, priorities, weights, gain_matrix, max_power):
        if method == 'Proportional Power':
            user_index = self.index
            optimal_power = self.proportional_power_algorithm(user_index, priorities, weights, gain_matrix, max_power)
            self.power = optimal_power
        else:
            raise ValueError("Invalid method. Must be 'Proportional Power'")
    
    def calculate_distance_to_rsu(self, rsu):
        """
        Calculates the Euclidean distance from the agent to a Road-Side Unit (RSU).

        Parameters:
        - rsu (RSU): The RSU object to calculate the distance to.

        Returns:
        - distance (float): The calculated distance to the RSU.
        """
        agent_position = torch.tensor(self.position, dtype=torch.float32)
        rsu_position = torch.tensor(rsu.position, dtype=torch.float32)
        distance = torch.norm(agent_position - rsu_position)
        self.distance = distance.item()  # Ensure distance is a scalar
        return self.distance

    def calculate_channel_gain(self, distance):
        """
        Calculates the channel gain for the agent using a path-loss model.

        Parameters:
        - distance (float): The distance to the RSU or communication partner in meters.

        Returns:
        - channel_gain (float): The calculated channel gain.
        """
        # Path-loss model parameters
        path_loss_exponent = 2.0  
        reference_distance = 1.0  # in meters
        reference_gain = 1.0       # Normalized reference gain
        min_gain = 0.01            # Minimum channel gain to prevent zero gain
        max_gain = 100.0           # Maximum channel gain for zero or invalid distance
        
        if isinstance(distance, (int, float)) and distance > 0:
            channel_gain = reference_gain * (reference_distance / distance) ** path_loss_exponent
            channel_gain = max(channel_gain, min_gain)
        else:
            # Handle zero or invalid distance
            channel_gain = max_gain
            logging.warning(f"Agent {self.name}: Invalid distance ({distance}). Assigning max_gain ({max_gain}).")
        
        self.channel_gain = channel_gain
        return channel_gain


    def calculate_snr(self, channel_gain):
        """
        Calculates the Signal-to-Noise Ratio (SNR) in decibels (dB) for the agent.

        Parameters:
        - channel_gain (float): The channel gain to use in the calculation.

        Returns:
        - snr_db (float): The calculated Signal-to-Noise Ratio in dB.
        """
        N0 = self.noise_level  # Noise power in Watts
        snr_linear = (self.power * channel_gain) / N0
        snr_db = np.log10(snr_linear) if snr_linear > 0 else -np.inf
        self.snr = snr_db
        return snr_db

    def calculate_rate(self, snr):
        """
        Calculates the data transmission rate using the Shannon-Hartley theorem.

        The rate is calculated as: Rate = Bandwidth * log2(1 + SNR).

        Parameters:
        - snr (float): The Signal-to-Noise Ratio to use in the calculation.

        Returns:
        - rate (float): The calculated data transmission rate.

        Raises:
        - ValueError: If no communication channel has been assigned to the agent.
        """
        if self.assigned_channel is None:
            self.assigned_channel = Channel(frequency=2.4, bandwidth=20)  # Removed trailing comma
        if snr <= 0:
            print(f"[WARNING] Agent {self.name} - Non-positive SNR ({snr}), setting rate to zero.")
            self.rate = 0
            return 0
        rate = self.assigned_channel.bandwidth * np.log2(1 + snr)
        self.rate = rate
        return rate

    def update_max_rate(self):
        # Direct transmission
        direct_snr = self.calculate_snr(self.distance)
        direct_rate = self.calculate_rate(direct_snr)
    
        # Indirect transmission via leader
        if self.leader:
            leader_distance = self.calculate_distance_to_agent(self.leader)
            penalty_ratio = 0.95 if leader_distance < 50 else 0.85
            indirect_snr = self.leader.calculate_snr(self.leader.distance) * penalty_ratio
            indirect_rate = self.calculate_rate(indirect_snr)
        else:
            indirect_rate = 0
    
        # Indirect without leader
        penalty_without_leader = 0.95 if self.distance < 100 else 0.8
        indirect_no_leader_snr = self.snr * penalty_without_leader
        indirect_no_leader_rate = self.calculate_rate(indirect_no_leader_snr)
    
        # Choose the best mode
        rates = {
            'Direct': direct_rate,
            'Indirect via Leader': indirect_rate,
            'Indirect without Leader': indirect_no_leader_rate
        }
        self.max_rate = max(rates.values())
        max_modes = [mode for mode, rate in rates.items() if rate == self.max_rate]
        self.transmission_mode = max_modes[0] if max_modes else None


    def perform_task(self, task):
        """
        Performs a specified task and returns the utility value.

        Parameters:
        - task (str): The task to perform ('transmit' or 'measure').

        Returns:
        - utility (float): The utility value obtained from performing the task.

        Tasks:
        - 'transmit': Utility is based on the data transmission rate.
        - 'measure': Utility is based on the Signal-to-Noise Ratio.
        """
        if task == 'transmit':
            return self.rate  # Utility based on rate
        elif task == 'measure':
            return self.snr  # Utility based on SNR
        else:
            print(f"[WARNING] Agent {self.name} - Unknown task '{task}'")
            return 0

    def calculate_distance_to_agent(self, other_agent):
        """
        Calculates the Euclidean distance to another agent.

        Parameters:
        - other_agent (CarAgent): The other agent to calculate the distance to.

        Returns:
        - distance (float): The calculated distance to the other agent.
        """
        self_position = torch.tensor(self.position, dtype=torch.float32)
        other_position = torch.tensor(other_agent.position, dtype=torch.float32)
        distance = torch.norm(self_position - other_position)
    
        return distance.item()

class IntersectionEnvironment:
    def compute_reward(self, actions, bandwidths, data_rates, T_max):
        """
        Computes the reward based on the agents' actions, their latency, bandwidth usage, and data transmission rates.

        Parameters:
        - actions: A tensor of actions taken by the agents.
        - bandwidths: Tensor representing the bandwidth allocated to each agent.
        - data_rates: Tensor representing the data rates achieved by each agent.
        - T_max: A scalar tensor or float representing the maximum acceptable latency.

        Returns:
        - clipped_reward: The total reward for the step, clipped to maintain stability.
        """

        # Calculate latency-related penalties
        normalized_latency = torch.sum(actions) / 1000
        latency_penalty = -torch.pow((normalized_latency / T_max), 2) if T_max != 0 else -normalized_latency

        # Calculate throughput reward
        throughput_reward = torch.sum(bandwidths * data_rates)

        # Combine penalties and rewards to compute total reward
        total_reward = latency_penalty + throughput_reward

        # Clip the reward to prevent extreme values which could destabilize training
        clipped_reward = torch.clamp(total_reward, -100, 100)

        return clipped_reward.item()

class IoVScenario(BaseScenario):
    """
    Represents a scenario for Intelligent Vehicle Operations (IoV) where multiple vehicles, 
    defined by platoons and individual cars, operate in a simulated environment with specific
    communication channels. This scenario is designed to simulate and analyze the behaviors 
    and interactions of these vehicles under various traffic conditions and communication constraints.

    Attributes:
        num_platoons (int): The number of platoons in the scenario.
        cars_per_platoon (int): The number of cars in each platoon.
        channels (list): A list of communication channels available for the vehicles.
        width (float): The width of the scenario area or lanes, derived from kwargs.
        length (float): The length of the scenario area or lanes, derived from kwargs.
        max_steering_angle (torch.Tensor): The maximum steering angle for vehicle dynamics, in radians.
        max_speed (float): The maximum speed limit for vehicles in the scenario.
        intersection (IntersectionEnvironment): A simulated intersection environment where vehicle interactions occur.
        agents (list): A list of agents (vehicles) that are active within the scenario.
        dataset (DataFrame): A dataset loaded from a CSV file that contains initial configurations or data for the vehicles.

    Parameters:
        num_platoons (int): Number of platoons in the simulation.
        cars_per_platoon (int): Number of cars per platoon.
        channels (list): Communication channels available to the vehicles.
        dataset_path (str): Path to the CSV file containing vehicle data.
        kwargs (dict): Additional keyword arguments for scenario dimensions or other settings.
    """
    def __init__(self, num_agents, cars_per_platoon, channels, dataset, **kwargs):
        super().__init__(**kwargs)
        self.num_agents = num_agents
        self.cars_per_platoon = cars_per_platoon
        self.channels = channels
        self.intersection = IntersectionEnvironment()
        self.agents = []
        self.dataset = dataset

    def make_world(self, batch_dim, device, **kwargs):
        world = World(batch_dim, device, substeps=10, collision_force=500)
        directions = ['north', 'south', 'east', 'west']
        direction_angles = {
            'north': 90,  # North should have a direction angle of 90 degrees
            'south': 270,  # South - 270 degrees
            'east': 0,  # East - 0 degrees
            'west': 180  # West - 180 degrees
        }

        agent_index = 0
        agent_data = self.dataset.groupby('Vehicle ID')
        for p, data in agent_data:
            leader = None
            for index, row in data.iterrows():
                direction = directions[agent_index % len(directions)]
                direction_angle = direction_angles[direction]
                agent = CarAgent(
                    index=agent_index,
                    position=[0,0],
                    speed=row['Speed (km/h)'] / 3600.0,  # Convert speed from km/h to m/s
                    direction_angle=direction_angle,  # Set the direction angle based on the movement direction
                    name=str(row['Vehicle ID']),
                    power=row['CPU Power (GHz)'],
                    snr=None,
                    rate=None,
                    channels=self.channels,
                    channel_gain=None,
                    leader=leader if index > 0 else None,
                    distance=None,
                    world=world,
                    direction=direction
                )
                if index == 0:
                    leader = agent  # Set the first agent as the leader if applicable
                world.add_agent(agent)
                self.agents.append(agent)
                agent_index += 1
        return world

    def reset_world_at(self, env_index=None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=0.1,
            x_bounds=(-1, 1),
            y_bounds=(-1, 1),
        )

    def observation(self, agent):
        return torch.cat([agent.state.pos, agent.state.vel], dim=-1)

    def reward(self, agent):
        actions = self.collect_actions()
        return self.intersection.compute_reward(actions, agent)
    
    def extra_render(self, env_index: int = 0):
        geoms = []
    
        # Define parameters for the squares and roads
        square_size = 1.5 # Half-size of the total area
        gap_size = 1.5     # Size of the gap in the center (intersection area)
        road_width = 0.4  # Width of the roads (adjusted for better visibility)
        dash_length = 0.1
        gap_length = 0.06
        rsu_radius = 0.25
    
        # Define the center position
        center_x = 0
        center_y = 0
    
        # Viewer settings
        viewer_size = [1200, 800]  # Example dimensions, adjust as needed
        bottom_margin = 700  # Margin from the bottom of the viewer
        line_height = 20  # Height of each text line
    
        # Initialize y-offset from the bottom of the screen
        y_offset = bottom_margin
        arm_length = 0.5  
        # Calculate the edges of the area
        left_x = center_x - square_size
        right_x = center_x + square_size
        top_y = center_y + square_size
        bottom_y = center_y - square_size
    
        gap_left_x = center_x - gap_size / 2
        gap_right_x = center_x + gap_size / 2
        gap_top_y = center_y + gap_size / 2
        gap_bottom_y = center_y - gap_size / 2
    
        # Draw the four squares (representing buildings or areas around the intersection)
        # Top-left square
        top_left_square = [
            (left_x, gap_top_y),
            (left_x, top_y),
            (gap_left_x, top_y),
            (gap_left_x, gap_top_y)
        ]
        geom = rendering.FilledPolygon(top_left_square)
        geom.set_color(0.7, 0.7, 0.7)  # Gray color for buildings
        geoms.append(geom)
    
        # Top-right square
        top_right_square = [
            (gap_right_x, gap_top_y),
            (gap_right_x, top_y),
            (right_x, top_y),
            (right_x, gap_top_y)
        ]
        geom = rendering.FilledPolygon(top_right_square)
        geom.set_color(0.7, 0.7, 0.7)
        geoms.append(geom)
    
        # Bottom-left square
        bottom_left_square = [
            (left_x, bottom_y),
            (left_x, gap_bottom_y),
            (gap_left_x, gap_bottom_y),
            (gap_left_x, bottom_y)
        ]
        geom = rendering.FilledPolygon(bottom_left_square)
        geom.set_color(0.7, 0.7, 0.7)
        geoms.append(geom)
    
        # Bottom-right square
        bottom_right_square = [
            (gap_right_x, bottom_y),
            (gap_right_x, gap_bottom_y),
            (right_x, gap_bottom_y),
            (right_x, bottom_y)
        ]
        geom = rendering.FilledPolygon(bottom_right_square)
        geom.set_color(0.7, 0.7, 0.7)
        geoms.append(geom)
        
        # Gather information to display (keeping your original code)
        for agent in self.agents:
            if hasattr(agent, 'channel_gain') and agent.channel_gain is not None:
                # Prepare the text to display
                info_text = (
                    f"{agent.name}:G {agent.channel_gain:.2f},"
                    f"SNR {agent.snr:.2f} dB, Rate {agent.rate:.2f} Mbps, "
                    f"Dist {agent.distance:.2f} m, "
                    f"P {agent.power:.2f} Watts, Ch {agent.assigned_channel.frequency} GHz,"
                    f"Mode: {agent.transmission_mode}"
                )
    
                # Create text geometry
                geom = rendering.TextLine(
                    text=info_text,
                    x=5,  # Margin from the left side
                    y=viewer_size[1] - y_offset,  # Position from the bottom
                    font_size=10
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geoms.append(geom)
                # Increment the y-offset for the next line of text
                y_offset += line_height
    
        # Draw dashed lines for roads in four directions
    
        # Upward direction (positive y-axis)
        road_length = top_y - gap_top_y
        num_dashes = int(road_length / (dash_length + gap_length))
        for i in range(num_dashes):
            start_y = gap_top_y + i * (dash_length + gap_length)
            end_y = start_y + dash_length
            if end_y > top_y:
                end_y = top_y
            dash = rendering.Line(
                start=(center_x, start_y),
                end=(center_x, end_y)
            )
            dash.set_linewidth(road_width)
            dash.set_color(0.5, 0.5, 0.5)  # Gray color
            geoms.append(dash)
    
        # Downward direction (negative y-axis)
        road_length = gap_bottom_y - bottom_y
        num_dashes = int(road_length / (dash_length + gap_length))
        for i in range(num_dashes):
            start_y = gap_bottom_y - i * (dash_length + gap_length)
            end_y = start_y - dash_length
            if end_y < bottom_y:
                end_y = bottom_y
            dash = rendering.Line(
                start=(center_x, start_y),
                end=(center_x, end_y)
            )
            dash.set_linewidth(road_width)
            dash.set_color(0.5, 0.5, 0.5)
            geoms.append(dash)
    
        # Rightward direction (positive x-axis)
        road_length = right_x - gap_right_x
        num_dashes = int(road_length / (dash_length + gap_length))
        for i in range(num_dashes):
            start_x = gap_right_x + i * (dash_length + gap_length)
            end_x = start_x + dash_length
            if end_x > right_x:
                end_x = right_x
            dash = rendering.Line(
                start=(start_x, center_y),
                end=(end_x, center_y)
            )
            dash.set_linewidth(road_width)
            dash.set_color(0.5, 0.5, 0.5)
            geoms.append(dash)
    
        # Leftward direction (negative x-axis)
        road_length = gap_left_x - left_x
        num_dashes = int(road_length / (dash_length + gap_length))
        for i in range(num_dashes):
            start_x = gap_left_x - i * (dash_length + gap_length)
            end_x = start_x - dash_length
            if end_x < left_x:
                end_x = left_x
            dash = rendering.Line(
                start=(start_x, center_y),
                end=(end_x, center_y)
            )
            dash.set_linewidth(road_width)
            dash.set_color(0.5, 0.5, 0.5)
            geoms.append(dash)
    
        # RSU at top-left corner
        rsu_circle = rendering.make_circle(radius=rsu_radius, filled=True)
        rsu_transform = rendering.Transform(
            translation=(center_x - square_size / 2.5 - arm_length, center_y + square_size / 2.5 + arm_length)
        )
        rsu_circle.add_attr(rsu_transform)
        rsu_circle.set_color(0.8, 0, 0)  # Red color for RSU
        geoms.append(rsu_circle)

        rsu_text = rendering.TextLine(
            text="RSU",
            x=65,
            y=600,
            font_size=12
        )
        rsu_text_transform = rendering.Transform()
        rsu_text.add_attr(rsu_text_transform)
        geoms.append(rsu_text)
    
        # Optionally keep or adjust the L shapes (commented out if not needed)
        # If you want to keep the L shapes, you can adjust their positions accordingly
    
        return geoms

class Channel:
    def __init__(self, frequency, bandwidth, priority=None):
        self.frequency = frequency
        self.bandwidth = bandwidth
        self.users = []  # This will keep track of active users (user indices or identifiers)
        self.priority = priority  # Priority attribute, can be dynamically assigned or fixed

    def add_user(self, user):
        """Add a user to the channel."""
        if user not in self.users:
            self.users.append(user)

    def remove_user(self, user):
        """Remove a user from the channel."""
        if user in self.users:
            self.users.remove(user)

    def get_current_load(self):
        """
        Returns the current load of the channel, defined here as the number of active users.
        """
        return len(self.users)  # Returns the count of active users

    def __str__(self):
        return f"Channel {self.frequency} GHz, Bandwidth {self.bandwidth} MHz, Priority {self.priority}, Current Load {self.get_current_load()} users"


class IoVEnvironment:
    def __init__(self, dataframe, state_dim, action_dim, num_agents, num_platoons, cars_per_platoon, channels, rsus,T_max):
        """
        Initializes the environment with agents, traffic lights, and other properties using dataset.
        """
        self.dataframe = dataframe
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.num_platoons = num_platoons
        self.cars_per_platoon = cars_per_platoon
        self.channels = channels
        self.rsus = rsus
        self.scenario = IoVScenario(self.num_agents, self.num_platoons, self.channels, self.dataframe)  
        self.environment = make_env(
            scenario=self.scenario, 
            num_envs=1,
            num_agents=self.num_agents,
            continuous=True,
            render=True
        )
        self.T_max = T_max
        
    def reset(self, rsu):
        """
        Resets the environment by resetting agent positions, speeds, and traffic light states.
        This function is called at the beginning of each new episode to reset the state.

        Returns:
        - state: A new random state for the environment.
        """
        
        # Reset agent positions and speeds
        for agent in self.scenario.agents:
            agent.position = (0, agent.position[1])  # Reset each agent's position along the x-axis
            agent.speed = 1  # Reset speed to the initial value

            # Assign a channel to the agent if it doesn't have one
            if agent.assigned_channel is None:
                agent.assigned_channel = self.channels[0]  # Assign the first channel as an example

        self.environment.reset()
        # Return a random state (can be changed to reflect more meaningful initial states)
        return np.random.randn(self.state_dim)

    def render(self):
        """
        Calls the VMAS environment to render the current state.
        """
        self.environment.render()  # This will trigger the 3D renderings

    def close(self):
        """
        Close the rendering window when done.
        """
        self.environment.close()
        
    def step(self, actions, channel_mode, power_mode):
        """
        Advances the environment by one time step based on the actions taken by the agents.
        This function applies the actions to the agents, updates traffic signals, and calculates the new state.
        It also handles communication aspects such as channel allocation, SNR calculation, and data rate computation.
        
        Parameters:
        - actions: A list of actions (e.g., speeds) taken by each agent.
        
        Returns:
        - next_state: The state of the environment after the step.
        - reward: The reward received based on the actions taken.
        - done: Boolean indicating if the episode has ended.
        """
        actions = torch.tensor(actions, dtype=torch.float32)
        
        bandwidths = torch.rand(len(self.scenario.agents))  # Placeholder initialization
        # Apply actions to each agent
        for agent, action in zip(self.environment.scenario.agents, actions):
            agent.speed = action  # Assuming each action represents a new speed
            agent.move()  # Update position based on new speed

        # Initialize priorities, weights, and gain_matrix for the power allocation algorithm
        priorities = [channel.priority for channel in self.channels]  # Replace with actual priority values
        weights = [channel.get_current_load() for channel in self.channels]  # Example method to determine current load
        gain_matrix = {(u.name, c): u.calculate_channel_gain(u.calculate_distance_to_rsu(self.rsus[0])) for u in self.scenario.agents for c in self.channels}
        max_power = 1
        
        for agent in self.scenario.agents:
            for rsu in self.rsus:
                if rsu.is_within_coverage(agent.position):
                    # Prepare data for the 'greedy' method
                    data_rates = {(a, c): a.calculate_rate(a.calculate_snr(a.calculate_channel_gain(a.calculate_distance_to_rsu(rsu)))) for a in self.scenario.agents for c in self.channels}
                    
                    agent.assign_channel(channel_mode, agent, self.channels, data_rates)
                    
                    # Pass all required arguments to allocate_power
                    agent.assign_power(
                        power_mode,
                        priorities=priorities,
                        weights=weights,
                        gain_matrix=gain_matrix,
                        max_power=max_power
                    )
                    # Update agent metrics based on newly assigned channel and power
                    distance = agent.calculate_distance_to_rsu(rsu)
                    channel_gain = agent.calculate_channel_gain(distance)
                    snr = agent.calculate_snr(channel_gain)
                    agent.rate = agent.calculate_rate(snr)
                    agent.update_max_rate()
                    agent.perform_task('transmit')

        # Calculate the new state of the environment
        next_state = self.get_state()  # This should compile the state from all agents
        data_rates = {agent.index: torch.rand(1) for agent in self.scenario.agents}
        data_rates = torch.cat(list(data_rates.values()))
        if bandwidths.shape != data_rates.shape:
            # Assuming both should have the shape of [self.num_agents], reshape or expand as necessary
            bandwidths = bandwidths.view(-1, 1)  # Reshape to [num_agents, 1]
            data_rates = data_rates.view(1, -1)  # Reshape to [1, num_agents]

        # Calculate the reward based on the actions
        reward = self.scenario.intersection.compute_reward(actions, bandwidths, data_rates, self.T_max)

        # Check if the episode has ended
        done = self.check_done()  # Define the conditions under which the episode ends

        return next_state, reward, done

    def get_state(self):
        """Compile the current state from all agents and traffic signals."""
        state = []
        # Gather agent states
        for agent in self.scenario.agents:
            state.extend(list(agent.position))  # Convert position (tuple) to list
            state.append(agent.speed)  # Add speed
        
        # Gather traffic signal states
        return np.array(state, dtype=np.float32)

   
    def check_done(self):
        """
        Determines if the current episode should end. The episode can end either randomly or when agents
        reach specific conditions (e.g., all agents have moved far enough).

        Returns:
        - done: Boolean indicating if the episode has ended.
        """
        # Randomly end episodes with a 5% chance to introduce variability
        done = np.random.rand() > 0.95
        
        # Check if all agents have moved beyond a certain distance (e.g., x > 10)
        all_agents_moved_far = all(agent.position[0] > 10 for agent in self.scenario.agents)
        
        # End the episode if all agents have moved far enough
        if all_agents_moved_far:
            done = True
        
        return done


def setup_environment(power_algo, channel_algo, mode, num_episodes=10000):
    """
    Sets up and runs the IoV environment simulation.

    Parameters:
    - power_algo (str): The power allocation algorithm to use ('max_power' or 'min_power').
    - channel_algo (str): The channel allocation method ('stable', 'greedy', 'wua').
    - mode (str): Operational mode ('manual' or 'auto').
    - num_episodes (int): Number of episodes to run in the simulation.
    """
    df = pd.read_csv('iov_dataset.csv')
    channels = [Channel(frequency=5.9, bandwidth=10), Channel(frequency=5, bandwidth=80), Channel(frequency=6, bandwidth=160)]

    center_x = 0  
    center_y = 0
    square_size = 1.5  
    gap_size = 1.5
    arm_length = 0.5  
    rsu_x = center_x - (square_size / 2 + arm_length)
    rsu_y = center_y + (square_size / 2 + arm_length)
    rsu_position = [rsu_x, rsu_y]

    rsus = [RSU(position=rsu_position, coverage_radius=5)]
    env = IoVEnvironment(df, state_dim=15, action_dim=10, num_agents=2, num_platoons=2, cars_per_platoon=2, channels=channels, rsus=rsus, T_max=100)
    replay_buffer = ReplayBuffer(capacity=100000)
    
    train_environment(df, env, rsus, num_episodes=num_episodes, channel_mode=channel_algo, replay_buffer=replay_buffer, power_mode=power_algo, operation_mode=mode)

def train_environment(df, env, rsus, num_episodes, channel_mode, replay_buffer, power_mode, operation_mode):
    """
    Train the IoV environment using specified parameters.
    
    Parameters are passed from the setup_environment function and are used to control the simulation.
    """
    average_latency = df['Latency (ms)'].mean()

    noise_processes = [GaussianNoise(action_dimension=action_dim) for _ in range(num_agents)]

    batch_size = 64
    warm_up = 1000
    update_target_every = 100
    gamma = 0.85
    tau = 0.01
    epsilon = 1.0

    episode_rewards = []
    episode_losses = []
    rsus = rsus

    for episode in range(num_episodes): 
        state = env.reset(rsus[0])
        total_reward = 0
        done = False
        
        while not done:
            # Render the environment for the current ti mestep (this will display the 3D window)
            env.render()
            time.sleep(0.08)
            # Your training logic (actions, next states, rewards, etc.)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).repeat(num_agents, 1)
            actions = np.array([actor(state_tensor[i].unsqueeze(0)).detach().cpu().numpy() for i, actor in enumerate(actors)])
            actions = actions.squeeze()  # Ensure actions are (num_agents, action_dim)
            
            # Introduce randomness with decreasing probability
            if np.random.rand() < epsilon:
                actions = np.random.uniform(-max_action, max_action, size=(num_agents, action_dim))
            
            noise = np.array([noise_process.noise() for noise_process in noise_processes])
            actions += noise  # Add noise to actions

            # Step the environment forward
            # next_state, reward, done = env.step(actions.sum(axis=0), channel_mode, power_mode, operation_mode)
            next_state, reward, done = env.step(actions.sum(axis=0), channel_mode, power_mode)
            replay_buffer.add(state, next_state, actions, reward, done)
            total_reward += reward

            if len(replay_buffer.buffer) > warm_up:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
                actor_loss, critic_loss = perform_updates(
                    batch_states,
                    batch_next_states,
                    batch_actions,
                    batch_rewards,
                    batch_dones,
                    actors,
                    target_actors,
                    critics,
                    target_critics,
                    optimizer_actors,
                    optimizer_critics,
                    gamma
                )
                episode_losses.append((actor_loss, critic_loss))  # Update with actual losses

            if episode % update_target_every == 0:
                for idx in range(num_agents):
                    soft_update(target_actors[idx], actors[idx], tau)

            state = next_state

        episode_rewards.append(total_reward)
        if episode % 100 == 0:
            avg_loss = np.nanmean(episode_losses) if episode_losses else float('nan')
            print(f"Episode: {episode}, Reward: {total_reward}, Average Loss: {avg_loss}")

    # Close the rendering window after training
    env.close()

    # Plotting results
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.subplot(122)
    plt.plot([np.mean(x) for x in episode_losses])
    plt.title('Average Loss per Episode')
    plt.show()


if __name__ == '__main__':
    # Example initialization from GUI input
    setup_environment(power_algo='Proportional Power', channel_algo='Greedy', mode='auto')