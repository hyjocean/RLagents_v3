import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

class ActorNet(nn.Module):
    def __init__(
            self,
            obs_space,
            act_space,
            config,
    ):
        super(ActorNet, self).__init__()
        self.device = config['device']
        # self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        # self._features = None
        n_input_channels = obs_space['maps'].shape[0]
        num_outputs = act_space.n
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (1,128,6,6)
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (1,128,4,4)
            nn.Conv2d(in_channels=512//4, out_channels=512-12, kernel_size=2, stride=1, padding="valid"),
            nn.Flatten(),
            nn.ReLU(),
        )

        self.goal_layer = nn.Linear(3, 12)

        self.hnn = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
        )

        self.hrelu = nn.ReLU()

        # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.cnn(th.as_tensor(obs_space['maps'].sample()[None]).float()).shape[1]

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, )
        
        # self.lstm._init_hidden(c_in, h_in)
        # self.state_in = (c_in, h_in)\

        self.policy_layer = nn.Linear(512, num_outputs) # 5 action_size
        self.policy_soft = nn.Softmax(dim=2)
        self.policy_sig = nn.Sigmoid()
        self.value_layer = nn.Linear(512, 1)
        self.blocking_layer = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.on_goal = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self._init_params()
        
    def _init_params(self):
        for ly in self.cnn:
            if hasattr(ly, 'weight'):
                nn.init.xavier_uniform_(ly.weight)  
            else:
                continue
        nn.init.xavier_uniform_(self.goal_layer.weight)
        for ly in self.hnn:
            if hasattr(ly, 'weight'):
                nn.init.xavier_uniform_(ly.weight)  
            else:
                continue
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                # 使用 Xavier 初始化权重
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                # 将偏置参数初始化为零
                nn.init.constant_(param, 0.0)

        nn.init.xavier_uniform_(self.policy_layer.weight)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.xavier_uniform_(self.blocking_layer[0].weight)
        nn.init.xavier_uniform_(self.on_goal[0].weight)

    # def forward(self, observations: torch.Tensor) -> torch.Tensor:
    def forward(self, maps, goal_vector, config, state=None):
        env_map_shape = config['env_map_shape']
        env_goal_shape = config['env_goal_shape']
        Batch_or_Agents_num = maps.shape[0]
        Seq_len_or_exp_num = maps.shape[1]

        maps = torch.tensor(np.array(maps)).float()# map.shape = [B/agents_num, Seq_len, 4, 10, 10]
        goal_vector = torch.tensor(np.array(goal_vector)).float() # goal_vector.shape = [B, Seq_len, 1, 3]
        
        # if maps.ndim == 3:
        #     maps = maps.unsqueeze(0)  # 这会在第0维度增加一个维度
        # if goal_vector.ndim == 2:
        #     goal_vector = goal_vector.unsqueeze(0)  # 这会在第0维度增加一个维度
        # print(maps)
        flat = self.cnn(maps.reshape(-1,*env_map_shape).to(self.device))
        goal_out = self.goal_layer(goal_vector.reshape(-1, *env_goal_shape).to(self.device))
        hidden_inp = torch.cat([flat.reshape(-1,500), goal_out.reshape(-1,12)], 1)
        hidden_state = self.hnn(hidden_inp)
        hidden_state = self.hrelu(hidden_state+hidden_inp)# [L, B, 512
        hidden_state = hidden_state.reshape(Batch_or_Agents_num, Seq_len_or_exp_num, 512)# [B, L, 512]
        # rnn_in = torch.unsqueeze(hidden_state, dim=1)
        # step_size = maps.shape[:1]
        state_in = [st.to(self.device) for st in state] if state is not None else self.get_initial_state(hidden_state.shape)
        # state_init = [x.permute(1,0,2).to(rnn_in.device) for x in state_in]
        lstm_out, lstm_state = self.lstm(hidden_state, state_in) # lstm_out.shape = [B, L, 512] hidden_state[0/1].shape = [1, B, 512]

        lstm_c, lstm_h = lstm_state
        state_out = [lstm_c, lstm_h]
        rnn_out = lstm_out 
        # rnn_out = lstm_out.reshape(-1,512)
        self._features = rnn_out

        policy_prob = self.policy_layer(rnn_out)
        policy = self.policy_soft(policy_prob)
        policy_sig = self.policy_sig(policy_prob)
        value = self.value_layer(rnn_out)
        blocking = self.blocking_layer(rnn_out)
        ongoal = self.on_goal(rnn_out)
        return policy, value, state_out, blocking, ongoal, policy_sig
    

    def get_initial_state(self,shape):
        B = shape[0]
        c_in = torch.zeros((1,B,512)).to(self.device)
        h_in = torch.zeros((1,B,512)).to(self.device)
        return [c_in, h_in]
    
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        x = self.value_layer(self._features)

        return torch.reshape(x, [-1])
        # if self.q_flag:
        #     return torch.reshape(self.vf_branch(x), [B, -1])
        # else:
        #     return torch.reshape(self.vf_branch(x), [-1])

    # def actor_parameters(self):
    #     return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    # def critic_parameters(self):
    #     return list(self.vf_branch.parameters())

    def metrics(self):
        # value_loss = torch.sum(train_value * torch.square())
        return {}
        # return {
        #     "policy_loss": self.policy_loss_metric,
        #     "imitation_loss": self.imitation_loss_metric,
        # }
    
    def custom_loss(self, policy_loss, loss_inputs):
        """Calculates a custom loss on top of the given policy_loss(es).

        Args:
            policy_loss (List[TensorType]): The list of already calculated
                policy losses (as many as there are optimizers).
            loss_inputs: Struct of np.ndarrays holding the
                entire train batch.

        Returns:
            List[TensorType]: The altered list of policy losses. In case the
                custom loss should have its own optimizer, make sure the
                returned list is one larger than the incoming policy_loss list.
                In case you simply want to mix in the custom loss into the
                already calculated policy losses, return a list of altered
                policy losses (as done in this example below).
        """
        

        # Get the next batch from our input files.
        # batch = reader.next()

        # # Define a secondary loss by building a graph copy with weight sharing.
        # obs = restore_original_dimensions(
        #     torch.from_numpy(batch["obs"]).float().to(policy_loss[0].device),
        #     self.obs_space,
        #     tensorlib="torch",
        # )
        # logits, _ = self.forward({"obs": obs}, [], None)

        # # Compute the IL loss.
        # action_dist = TorchCategorical(logits, self.model_config)
        # imitation_loss = torch.mean(
        #     -action_dist.logp(
        #         torch.from_numpy(batch["actions"]).to(policy_loss[0].device)
        #     )
        # )
        # self.imitation_loss_metric = imitation_loss.item()
        # self.policy_loss_metric = np.mean([loss.item() for loss in policy_loss])

        # # Add the imitation loss to each already calculated policy loss term.
        # # Alternatively (if custom loss has its own optimizer):
        # # return policy_loss + [10 * self.imitation_loss]
        # return [loss_ + 10 * imitation_loss for loss_ in policy_loss]
        return policy_loss 
    
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

class ActorNet1(nn.Module):
    def __init__(
            self,
            obs_space,
            act_space,
            config,
    ):
        super(ActorNet1, self).__init__()
        self.device = config['device']
        # self.full_obs_space = getattr(obs_space, "original_space", obs_space)
        # self._features = None
        n_input_channels = obs_space['maps'].shape[0]
        num_outputs = act_space.n
        
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (1,128,6,6)
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=512//4, out_channels=512//4, kernel_size=3, stride=1, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # (1,128,4,4)
            nn.Conv2d(in_channels=512//4, out_channels=512-12, kernel_size=2, stride=1, padding="valid"),
            nn.Flatten(),
            nn.ReLU(),
        )

        self.goal_layer = nn.Linear(3, 12)

        self.hnn = nn.Sequential(
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
        )

        self.hrelu = nn.ReLU()

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        # self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, )
        # self.lstm._init_hidden(c_in, h_in)
        # self.state_in = (c_in, h_in)\

        self.policy_layer = nn.Linear(512, num_outputs) # 5 action_size
        self.policy_soft = nn.Softmax(dim=2)
        self.policy_sig = nn.Sigmoid()
        self.value_layer = nn.Linear(512, 1)
        self.blocking_layer = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.on_goal = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self._init_params()
        
    def _init_params(self):
        for ly in self.cnn:
            if hasattr(ly, 'weight'):
                nn.init.xavier_uniform_(ly.weight)  
            else:
                continue
        nn.init.xavier_uniform_(self.goal_layer.weight)
        for ly in self.hnn:
            if hasattr(ly, 'weight'):
                nn.init.xavier_uniform_(ly.weight)  
            else:
                continue
        # for name, param in self.lstm.named_parameters():
        #     if 'weight' in name:
        #         # 使用 Xavier 初始化权重
        #         nn.init.xavier_normal_(param)
        #     elif 'bias' in name:
        #         # 将偏置参数初始化为零
        #         nn.init.constant_(param, 0.0)

        nn.init.xavier_uniform_(self.policy_layer.weight)
        nn.init.xavier_uniform_(self.value_layer.weight)
        nn.init.xavier_uniform_(self.blocking_layer[0].weight)
        nn.init.xavier_uniform_(self.on_goal[0].weight)

    # def forward(self, observations: torch.Tensor) -> torch.Tensor:
    def forward(self, maps, goal_vector, config, state=None):
        env_map_shape       = config['env_map_shape']
        env_goal_shape      = config['env_goal_shape']
        Batch_or_Agents_num = maps.shape[0]
        Seq_len_or_exp_num  = maps.shape[1]

        maps = torch.tensor(np.array(maps)).float()# map.shape = [B/agents_num, Seq_len, 4, 10, 10]
        goal_vector = torch.tensor(np.array(goal_vector)).float() # goal_vector.shape = [B, Seq_len, 1, 3]
        
        flat = self.cnn(maps.reshape(-1,*env_map_shape).to(self.device))
        goal_out = self.goal_layer(goal_vector.reshape(-1, *env_goal_shape).to(self.device))
        hidden_inp = torch.cat([flat.reshape(-1,500), goal_out.reshape(-1,12)], 1)
        hidden_state = self.hnn(hidden_inp)
        hidden_state = self.hrelu(hidden_state+hidden_inp)# [L, B, 512
        hidden_state = hidden_state.reshape(Batch_or_Agents_num, Seq_len_or_exp_num, 512)# [B, L, 512]
        # rnn_in = torch.unsqueeze(hidden_state, dim=1)
        # step_size = maps.shape[:1]
        # state_in = [st.to(self.device) for st in state] if state is not None else self.get_initial_state(hidden_state.shape)
        # state_init = [x.permute(1,0,2).to(rnn_in.device) for x in state_in]
        # lstm_out, lstm_state = self.lstm(hidden_state, state_in) # lstm_out.shape = [B, L, 512] hidden_state[0/1].shape = [1, B, 512]

        # lstm_c, lstm_h = lstm_state
        state_out = hidden_state
        # rnn_out = lstm_out 
        # rnn_out = lstm_out.reshape(-1,512)
        self._features = hidden_state

        policy_prob = self.policy_layer(hidden_state)
        policy = self.policy_soft(policy_prob)
        policy_sig = self.policy_sig(policy_prob)
        value = self.value_layer(hidden_state)
        blocking = self.blocking_layer(hidden_state)
        ongoal = self.on_goal(hidden_state)
        return policy, value, state_out, blocking, ongoal, policy_sig
    

    def get_initial_state(self,shape):
        B = shape[0]
        c_in = torch.zeros((1,B,512)).to(self.device)
        h_in = torch.zeros((1,B,512)).to(self.device)
        return [c_in, h_in]
    
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        x = self.value_layer(self._features)

        return torch.reshape(x, [-1])
        # if self.q_flag:
        #     return torch.reshape(self.vf_branch(x), [B, -1])
        # else:
        #     return torch.reshape(self.vf_branch(x), [-1])
