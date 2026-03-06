import torch
from lstm_model import RobotDynamicsLSTM

# Create the model
model = RobotDynamicsLSTM()
model.reset_hidden()

# Dummy input: current state and control [x, y, θ, v, ω]
# Suppose: x=1.0, y=2.0, θ=0.5 rad, v=0.2 m/s, ω=0.1 rad/s
test_input = torch.tensor([[[1.0, 2.0, 0.5, 0.2, 0.1]]], dtype=torch.float32)  # shape: (1, 1, 5)

# Predict next state
with torch.no_grad():  # no training yet
    pred = model(test_input)

print("Predicted next state (x, y, θ):", pred.numpy())
