import torch
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
from web3 import Web3
from model_architecture import NeuralNetwork


# Function to visualize data distribution
def getDist(y):
    plt.hist(y, bins=np.arange(11) - 0.5, rwidth=0.8)
    plt.title("Count of data classes")
    plt.xticks(np.arange(10))
    plt.show()

# Function to get data based on distribution
def getData(dist, dataset):
    targets = np.array(dataset.targets)
    indices = []
    
    for i in range(10):  # For each class
        class_indices = np.where(targets == i)[0]
        selected_indices = class_indices[:dist[i]]
        indices.extend(selected_indices)
    
    subset = torch.utils.data.Subset(dataset, indices)
    return subset

# Load in MNIST dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor())
print(len(mnist_trainset))
print(len(mnist_testset))

model = NeuralNetwork()

# Distributions of each number samples
first_model_distributions = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]    # Client 1 - the aim is to have a very bad model to try improve upon

first_model_subset = getData(first_model_distributions, mnist_trainset)

first_model_train_loader = DataLoader(first_model_subset, batch_size=64, shuffle=True)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Calculate metrics (precision, recall and F1 score)
def calculate_metrics(predicted, targets, num_classes=10):
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1_score = torch.zeros(num_classes)

    for i in range(num_classes):
        True_positives = ((predicted == i) & (targets == i)).float().sum()
        false_positives = ((predicted == i) & (targets != i)).float().sum()
        false_negatives = ((predicted != i) & (targets == i)).float().sum()

        if True_positives == 0:
            precision[i], recall[i], f1_score[i] = 0, 0, 0
        else:
            precision[i] = True_positives / (True_positives + false_positives)
            recall[i] = True_positives / (True_positives + false_negatives)
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    return precision, recall, f1_score

# Display the metrics clearly in the terminal
def print_metrics_table(precision, recall, f1_score, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(len(precision))]
    header = f"{'Class':^10} | {'Precision':^10} | {'Recall':^10} | {'F1-Score':^10}"
    print(header)
    print("-" * len(header))
    for i, (p, r, f1) in enumerate(zip(precision, recall, f1_score)):
        class_name = class_names[i] if class_names else str(i)
        print(f"{class_name:^10} | {p:^10.4f} | {r:^10.4f} | {f1:^10.4f}")

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # For MNIST dataset

# Training the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for images, labels in first_model_train_loader:
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(first_model_train_loader)}")

print("Training completed.")

# Evaluating the model
model.eval()
total_predicted = torch.tensor([]).long()
total_targets = torch.tensor([]).long()

with torch.no_grad():
    for images, labels in DataLoader(mnist_testset, batch_size=64):
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total_predicted = torch.cat((total_predicted, predicted))
        total_targets = torch.cat((total_targets, labels))

precision, recall, f1_score = calculate_metrics(total_predicted, total_targets)
print_metrics_table(precision, recall, f1_score, class_names)

# Saving the model

# Formats the datetime as: YYYYMMDD-HHMMSS (e.g., 20230329-153045)
datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

model_path = f"./saved_models/mnist_neural_network_{datetime_str}.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


# Connect to Ganache
ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))
web3.is_connected()  # Should return True

# Set up the contract
# contract_address = web3.to_checksum_address("0xfb66B39B74a6e2059725efd4411780C90749636C")
contract_address = web3.to_checksum_address("0x71Ec77cdD098cEB33Ab073a5829C03E1e4125eb2")
contract_abi = [
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "name": "confusionMatrix",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [],
      "name": "fileName",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [],
      "name": "lastUpdated",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [],
      "name": "versionCounter",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [
        {
          "internalType": "string",
          "name": "_fileName",
          "type": "string"
        },
        {
          "internalType": "string",
          "name": "_dateTime",
          "type": "string"
        }
      ],
      "name": "setfileName",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [],
      "name": "getfileName",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [],
      "name": "getLastUpdated",
      "outputs": [
        {
          "internalType": "string",
          "name": "",
          "type": "string"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [],
      "name": "getVersionCounter",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [],
      "name": "initializeConfusionMatrix",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "row",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "col",
          "type": "uint256"
        }
      ],
      "name": "getConfusionMatrixValue",
      "outputs": [
        {
          "internalType": "uint256",
          "name": "",
          "type": "uint256"
        }
      ],
      "stateMutability": "view",
      "type": "function",
      "constant": true
    },
    {
      "inputs": [
        {
          "internalType": "uint256",
          "name": "row",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "col",
          "type": "uint256"
        },
        {
          "internalType": "uint256",
          "name": "value",
          "type": "uint256"
        }
      ],
      "name": "updateConfusionMatrix",
      "outputs": [],
      "stateMutability": "nonpayable",
      "type": "function"
    }
  ]

contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Assuming you have the account unlocked which is sending the transactions
from_address = web3.eth.accounts[0]

# Interact with the contract
tx_hash = contract.functions.setFileHash(f"./saved_models/mnist_neural_network_{datetime_str}.pth").transact({'from': from_address})
web3.eth.wait_for_transaction_receipt(tx_hash)

# Retrieve the file hash from the contract
file_hash = contract.functions.getFileHash().call()
print(file_hash)