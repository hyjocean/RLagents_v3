import torch
import torch.nn as nn
import torch.optim as optim

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)

class GridCNN(nn.Module):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param obs_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """
    def __init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs,
    ):
        nn.Module.__init__(self)

        self.full_obs_space = getattr(obs_space, "original_space", obs_space)

        self._features = None
        n_input_channels = self.full_obs_space['maps'].shape[0]

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
            nn.Dropout(1),
            nn.Linear(512, 512),
            nn.Dropout(1),
        )

        self.hrelu = nn.ReLU()

        # Compute shape by doing one forward pass
        # with th.no_grad():
        #     n_flatten = self.cnn(th.as_tensor(obs_space['maps'].sample()[None]).float()).shape[1]

        # self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.lstm = nn.LSTM(input_size=512, hidden_size=512, batch_first=True, )
        
        # self.lstm._init_hidden(c_in, h_in)
        # self.state_in = (c_in, h_in)

        self.policy_layer = nn.Linear(512, num_outputs) # 5 action_size
        self.policy_soft = nn.Softmax()
        self.policy_sig = nn.Sigmoid()
        self.value_layer = nn.Linear(512, 1)
        self.blocking_layer = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.on_goal = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        
        
    # def forward(self, observations: torch.Tensor) -> torch.Tensor:
    def forward(self, input_dict:
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        with open('/home/bld/HK_RL/RLagents_ptz/tmp_log.txt', 'w') as file:
            # 在文件末尾追加信息
            file.write(f"seq_lens:,{seq_lens}, batch: {input_dict['obs']['maps'].shape}, state: {state[0].shape}\n")


        # if len(state[0].shape)!=3:
        #     state = [x.unsqueeze(1) for x in state]
            
        maps = input_dict['obs']['maps']
        goal_vector = input_dict['obs']['goal_vector']
        # goal_vector = input_dict['obs'][:,:3].reshape(-1,3)
        # maps = input_dict['obs'][:,3:].reshape(-1, *(self.full_obs_space['obs']['maps'].shape))
        self.time_major = self.model_config.get("_time_major", False)
        if isinstance(seq_lens, np.ndarray) or isinstance(seq_lens, torch.Tensor):
            seq_lens = torch.Tensor(seq_lens).int() if seq_lens.dtype != torch.int32 else seq_lens
            max_seq_len = maps.shape[0] // seq_lens.shape[0]
            maps = add_time_dimension(
                maps,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=self.time_major,
            )
            goal_vector = add_time_dimension(
                goal_vector,
                max_seq_len=max_seq_len,
                framework="torch",
                time_major=self.time_major,
            )
        
        if len(maps.shape) == 5:
            maps=maps.reshape(-1, *(maps.shape[2:]))
            goal_vector=goal_vector.reshape(-1, *(goal_vector.shape[2:]))
            state = [x.repeat(max(seq_lens),1,1) for x in state] 
        flat = self.cnn(maps.squeeze(1) if len(maps.shape)==5 else maps)
        # print('flat:', flat.shape)
        goal_layer = self.goal_layer(goal_vector.squeeze(1))
        hidden_inp = torch.cat([flat.reshape(-1,500), goal_layer.reshape(-1,12)], 1)
        hidden_state = self.hnn(hidden_inp)
        hidden_state = self.hrelu(hidden_state+hidden_inp)
        
        rnn_in = torch.unsqueeze(hidden_state, dim=1)
        step_size = maps.shape[:1]
        # c_in = torch.zeros((1, 1, 512))
        # h_in = torch.zeros((1, 1, 512))
        # state_init = (c_in.to(rnn_in.device), h_in.to(rnn_in.device))
        state_init = [x.permute(1,0,2).to(rnn_in.device) for x in state]
        lstm_out, lstm_state = self.lstm(rnn_in, state_init)

        lstm_c, lstm_h = lstm_state
        # state_out = [lstm_c.squeeze(0), lstm_h.squeeze(0)]
        state_out = [lstm_c.permute(1,0,2), lstm_h.permute(1,0,2)]
        rnn_out = lstm_out.reshape(-1,512)
        self._features = rnn_out

        policy_prob = self.policy_layer(rnn_out)
        # policy = self.policy_soft(policy_prob)
        policy_sig = self.policy_sig(policy_prob)
        value = self.value_layer(rnn_out)
        blocking = self.blocking_layer(rnn_out)
        ongoal = self.on_goal(rnn_out)
        return policy_prob, state_out
    

    def get_initial_state(self):
        c_in = torch.zeros((1, 512))
        h_in = torch.zeros((1, 512))
        return [c_in, h_in]
    
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        B = self._features.shape[0]
        x = self.value_layer(self._features)

        return torch.reshape(x, [-1])
        # if self.q_flag:
        #     return torch.reshape(self.vf_branch(x), [B, -1])
        # else:
        #     return torch.reshape(self.vf_branch(x), [-1])

    def actor_parameters(self):
        return reduce(lambda x, y: x + y, map(lambda p: list(p.parameters()), self.actors))

    def critic_parameters(self):
        return list(self.vf_branch.parameters())

    def metrics(self):
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
    