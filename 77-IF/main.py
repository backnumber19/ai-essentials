# pip install --user pytorch-influence-functions
import torch
import torchvision
import pytorch_influence_functions as ptif
from torch.utils.data import DataLoader

# Load MNIST data
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_dataset = torchvision.datasets.MNIST(
    "./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST("./data", train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Define simple CNN model
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


model = SimpleCNN()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# ... training code omitted ...

# Calculate Influence Functions
ptif.init_logging()
config = ptif.get_default_config()
influences, harmful, helpful = ptif.calc_img_wise(
    config, model, train_loader, test_loader
)

print(f"Most helpful training samples: {helpful[:5]}")
print(f"Most harmful training samples: {harmful[:5]}")
