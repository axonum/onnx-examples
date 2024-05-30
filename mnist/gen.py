# MNIST example

import torch
import torch.onnx
from onnx import TensorProto

# fix random seed
torch.manual_seed(42)

# Define the model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(1600, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = torch.nn.functional.softmax(x, dim=1)
        return output

# Create the model
model = Net()
model.eval()

# Export the model
input = torch.randn(1, 1, 28, 28)
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