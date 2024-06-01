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
        self.conv1 = torch.nn.Conv2d(1, 8, 5, 1)
        self.conv2 = torch.nn.Conv2d(8, 16, 5, 1)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(256, 10)
    
    def forward(self, x):
        # cast to float32
        x = x.type(torch.float32)
        # normalize
        x = torch.div(x, 255.0)
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = torch.nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # softmax if training, argmax if inference
        if self.training:
            return torch.nn.functional.log_softmax(x, dim=1)
        else:
            return torch.argmax(x, dim=1, keepdim=False).type(torch.uint8)

# Create the model
model = Net()

# Train the model without normalization
import torchvision
import torchvision.transforms as transforms
# input as int64, do not normalize
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.PILToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train for 1 epoch
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

# Switch to eval mode
model.eval()

# Export the model with an example test input in int64
input = trainset[0][0]
# save input as image
transforms.functional.to_pil_image(input).save("data/input.png")
torch.onnx.export(model, input.unsqueeze(0), "model.onnx", export_params=True, opset_version=7, do_constant_folding=True, input_names = ['input'], output_names = ['output'], verbose=False)

# Export sample input and output
output = model(input.unsqueeze(0))

# save input.bin
input = input.detach().numpy()
print(input)
input.tofile("input.bin")
print("Input: 0x"+input.data.hex())

# save output.bin
output = output.detach().numpy()
output.tofile("output.bin")
print("Output: 0x"+output.data.hex())