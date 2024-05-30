# simple 10-dim linear regression model

import torch

# fix random seed
torch.manual_seed(42)

# Define the model
class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(10, 1)
    def forward(self, x):
        return self.linear(x)
    
# Create the model
model = LinearRegression()
model.eval()

# Export the model
input = torch.randn(1, 10)
torch.onnx.export(model, input, "model.onnx", export_params=True, opset_version=7, do_constant_folding=True, input_names = ['input'], output_names = ['output'], verbose=False)

# Export sample input and output
output = model(input)

# save input.bin
input = input.detach().numpy()
input.tofile("input.bin")
print("Input: 0x"+input.data.hex())

# save output.bin
output = output.detach().numpy()
output.tofile("output.bin")
print("Output: 0x"+output.data.hex())