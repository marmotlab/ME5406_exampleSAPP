import torch
import torch.nn as nn
import torch.nn.functional as F

# parameters for training
RNN_SIZE = 128
GOAL_REPR_SIZE = 12


class ACNet(nn.Module):
    def __init__(self, a_size, NUM_CHANNEL, GRID_SIZE):
        super(ACNet, self).__init__()
        self.state_shape = [NUM_CHANNEL, GRID_SIZE, GRID_SIZE]
        self.goal_shape = [3]
        self.input_channel = NUM_CHANNEL

        self.sequential_block = nn.Sequential(nn.Conv2d(self.input_channel, RNN_SIZE // 4, 3, 1, 1), nn.ReLU(),
                                              nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 4, 3, 1, 1), nn.ReLU(),
                                              nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 4, 3, 1, 1), nn.ReLU(),
                                              nn.MaxPool2d(2),
                                              nn.Conv2d(RNN_SIZE // 4, RNN_SIZE // 2, 3, 1, 1), nn.ReLU(),
                                              nn.Conv2d(RNN_SIZE // 2, RNN_SIZE // 2, 3, 1, 1), nn.ReLU(),
                                              nn.Conv2d(RNN_SIZE // 2, RNN_SIZE // 2, 3, 1, 1), nn.ReLU(),
                                              nn.MaxPool2d(2),
                                              nn.Conv2d(RNN_SIZE // 2, RNN_SIZE - GOAL_REPR_SIZE, 2, 1),
                                              nn.ReLU(),
                                              nn.Flatten())

        self.fully_connected_1 = nn.Sequential(nn.Linear(self.goal_shape[0], GOAL_REPR_SIZE), nn.ReLU())
        self.fully_connected_2 = nn.Sequential(nn.Linear(RNN_SIZE, RNN_SIZE), nn.ReLU())
        self.fully_connected_3 = nn.Linear(RNN_SIZE, RNN_SIZE)

        self.Lstm = nn.LSTM(input_size=RNN_SIZE, hidden_size=RNN_SIZE, num_layers=1, batch_first=True)

        self.policy_layer = nn.Linear(RNN_SIZE, a_size)
        torch.nn.init.xavier_normal_(self.policy_layer.weight)
        self.softmax_layer = nn.Softmax(dim=1)
        self.sigmoid_layer = nn.Sigmoid()
        self.value_layer = nn.Linear(RNN_SIZE, 1)

    def forward(self, inputs, goal_pos, rnn_state):
        flat = self.sequential_block(inputs)
        goal_layer = self.fully_connected_1(goal_pos)
        hidden_input = torch.concat([flat, goal_layer], 1)
        h1 = self.fully_connected_2(hidden_input)
        h2 = self.fully_connected_3(h1)
        h3 = F.relu(h2 + hidden_input)
        rnn_in = h3.unsqueeze(1)
        r_out, (h_n, c_n) = self.Lstm(rnn_in, rnn_state)
        r_out = r_out.view(-1, RNN_SIZE)
        # Policy layer and value layer
        policy_layer = self.policy_layer(r_out)
        policy = self.softmax_layer(policy_layer)
        policy_sig = self.sigmoid_layer(policy_layer)
        value = self.value_layer(h3)

        return policy, value, policy_sig, (h_n, c_n)
