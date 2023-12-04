import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# Hyperparameters
input_dim = 1024
hidden_dim = 2048
output_dim = 1024
learning_rate = 0.001
num_epochs = 50
batch_size = 32

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TranslationContexts(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


print("Load data from disk")
# Assuming X and Y are your data tensors
X = torch.from_numpy(np.load("/ivi/ilps/personal/dstap1/data/fairseq_ted_m2m_100/X_train.npy"))
Y = torch.from_numpy(np.load("/ivi/ilps/personal/dstap1/data/fairseq_ted_m2m_100/Y_train.npy"))
print("Finished loading")

dataset = TranslationContexts(X, Y)

# Determine the lengths of each split
total_size = len(dataset)
test_size = dev_size = int(total_size * 0.1)
train_size = total_size - 2 * dev_size

# Create the splits
train_dataset, dev_dataset, test_dataset = random_split(dataset, [train_size, dev_size, test_size])

# Create DataLoaders for each split
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create the network
net = Net(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

X_test, Y_test = torch.stack([x for x,_ in test_dataset]), torch.stack([y for _,y in test_dataset])


print("L2 distance (raw): ", torch.mean(torch.sqrt(torch.sum((X_test-Y_test)**2, axis=1))))

# Training loop
print("Start training")
best_dev_loss = float('inf')
steps_without_improvement = 0
best_model = None
for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(train_dataloader):
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # print(f"Step {i}; loss = {loss}")

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Evaluation step every 500 steps
        if (i+1) % 500 == 0 or i == 0:
            net.eval()
            with torch.no_grad():
                dev_loss = 0
                for inputs, targets in dev_dataloader:
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    dev_loss += loss.item()
                dev_loss /= len(dev_dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Train Loss: {loss.item()}, Dev Loss: {dev_loss}')
            
            # Check for improvement
            if dev_loss < best_dev_loss:
                print("New best model!")
                best_dev_loss = dev_loss
                steps_without_improvement = 0
                
                # Save the model
                best_model = net.state_dict()

                print("L2 distance (current model): ", torch.mean(torch.sqrt(torch.sum((net(X_test)-Y_test)**2, axis=1))))

            else:
                steps_without_improvement += 1
                
            # Stop training if there's no improvement for 10 steps
            if steps_without_improvement == 10:
                print('No improvement for 10 steps, stopping')
                break
            
            net.train()

    # Break the outer loop if we're done
    if steps_without_improvement == 10:
        break

print('Finished Training')






net.load_state_dict(best_model)
torch.save(net.state_dict(), 'best.pth')



