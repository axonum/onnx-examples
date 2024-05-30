# simple 10-dim logistic regression model and round to 0 or 1

import torch

# fix random seed
torch.manual_seed(42)

# Define the model
class LogisticRegression(torch.nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        x =  torch.sigmoid(self.linear(x))
        # Round the output to 0 or 1
        return torch.add(torch.mul(x >= 0.5, 1), torch.mul(x < 0.5, 0)).type(torch.int64)
    
# Create the model
model = LogisticRegression()
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