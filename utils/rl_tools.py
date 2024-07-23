from collections import deque
import scipy.signal as signal
import torch
import numpy as np
def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x, gamma)

class ReplayMemory:
    def __init__(self, capacity, device):
        """
        Experience Replay Memory defined by deques to store transitions/agent experiences
        """
        
        self.capacity = capacity
        self.device = device
        
        self.maps             = deque(maxlen=capacity)
        self.goal_vector      = deque(maxlen=capacity)
        self.next_maps        = deque(maxlen=capacity)
        self.next_goal_vector = deque(maxlen=capacity)
        self.actions          = deque(maxlen=capacity)
        self.rewards          = deque(maxlen=capacity)
        self.dones            = deque(maxlen=capacity)
        
        
    def store(self, state, next_state, reward, done):
        """
        Append (store) the transitions to their respective deques
        """
        
        self.maps.append(state['maps']) 
        self.goal_vector.append(state['goal_vector'])
        self.next_maps.append(next_state['maps'])
        self.next_goal_vector.append(next_state['goal_vector'])
        self.rewards.append(reward)
        self.dones.append(done)
        
        
    def sample(self, batch_size):
        """
        Randomly sample transitions from memory, then convert sampled transitions
        to tensors and move to device (CPU or GPU).
        """
        
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=self.device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=self.device) for i in indices]).to(self.device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=self.device)

        return states, actions, next_states, rewards, dones
    
    
    def __len__(self):
        """
        To check how many samples are stored in the memory. self.dones deque 
        represents the length of the entire memory.
        """
        
        return len(self.dones)
    