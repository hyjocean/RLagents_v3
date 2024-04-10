import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

def collect_rollouts(env, actor_net, critic_net, num_steps, imitation_data=None, use_imitation=False):
    states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

    state = env.reset()
    for _ in range(num_steps):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        if use_imitation and imitation_data:
            state, action = imitation_data.pop(0)
            log_prob = torch.log(torch.tensor(1.0)) 
        else:
            with torch.no_grad():
                probs = actor_net(state_tensor)
                action = np.random.choice(env.action_space.n, p=np.squeeze(probs.numpy()))
                log_prob = torch.log(probs.squeeze(0)[action])
                value = critic_net(state_tensor)
                next_state, reward, done, _ = env.step(action)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        values.append(value)
        dones.append(done)

        state = next_state if not done else env.reset()

    return states, actions, log_probs, rewards, values, dones
