import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from web3 import Web3
import json
import os

# Initialize Web3
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
contract_address = web3.to_checksum_address("0x88C3CE4777861c5C103E46A8B0849d1Ab1e27c11")

# Load the contract ABI
with open('FileStorageABI.json', 'r') as file:
    contract_abi = json.load(file)
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),  # MNIST images are 28x28
            nn.ReLU(),
            nn.Linear(512, 10)  # 10 classes for MNIST digits
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

def prepare_dataset(df):
    X = df.iloc[:, 1:].values / 255.0
    y = df.iloc[:, 0].values
    return TensorDataset(torch.Tensor(X), torch.Tensor(y).long())

def train_and_evaluate(model, train_loader, test_loader, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    cmatrix = confusion_matrix(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, zero_division=0)
    accuracy = np.trace(cmatrix) / np.sum(cmatrix) * 100
    return accuracy, cmatrix, precision, recall, f1_score


def fetch_all_client_data_paths():
    client_addresses = contract.functions.getAllClientAddresses().call()
    dataset_paths = []
    addresses = []
    for address in client_addresses:
        contributions = contract.functions.getAllContributionsForClient(address).call()
        for data in contributions:
            dataset_paths.append(data[1])  # Fetching dataset paths
            addresses.append(address)      # Store the corresponding address
    return dataset_paths, addresses


def update_blockchain_contributions(address, contributions, data_path):
    """
    Updates or adds new client contributions to the blockchain.
    :param address: Ethereum address of the client
    :param contributions: Array of contributions to update
    :param data_path: Path of the dataset used
    """
    # Fetch existing contributions for the client
    client_contributions = contract.functions.getAllContributionsForClient(address).call()
    if len(client_contributions) > 0:
        # Assume you want to update the first entry for simplicity
        contract.functions.updateClientContributions(address, 0, contributions).transact({'from': web3.eth.default_account})
    else:
        # Add new client data if no contributions exist yet
        contract.functions.addClientData(contributions, data_path).transact({'from': web3.eth.default_account})

# def calculate_reward_and_matrix():
#     dataset_paths, client_addresses = fetch_all_client_data_paths()
#     client_datasets = [prepare_dataset(pd.read_csv(path)) for path in dataset_paths if os.path.exists(path)]
#     test_df = pd.read_csv('mnist-test/mnist_test.csv')
#     test_dataset = prepare_dataset(test_df)
#     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

#     incremental_dataset = None
#     incremental_f1_scores = []

#     for i, client_data in enumerate(client_datasets):
#         if incremental_dataset is None:
#             incremental_dataset = client_data
#         else:
#             incremental_dataset = ConcatDataset([incremental_dataset, client_data])

#         train_loader = DataLoader(incremental_dataset, batch_size=64, shuffle=True)
#         model = NeuralNetwork()
#         accuracy, confusion, precision, recall, f1_score = train_and_evaluate(model, train_loader, test_loader)
#         incremental_f1_scores.append(f1_score)
#         final_confusion_matrix = confusion

#         print(f'Incremental accuracy after adding client {i+1}: {accuracy:.2f}%')
#         print("Metrics per class:")
#         print(pd.DataFrame({
#             'Precision': precision,
#             'Recall': recall,
#             'F1 Score': f1_score
#         }, index=[f'Class {j}' for j in range(len(precision))]))
#         print("\n")

#     # Calculate the total improvement for each class from the initial to the final client
#     f1_df = pd.DataFrame(incremental_f1_scores, columns=[f'Class {i}' for i in range(10)])
#     f1_df['Client'] = [f'Client {i+1}' for i in range(len(client_datasets))]
#     f1_df.set_index('Client', inplace=True)

#     initial_f1 = np.zeros(10)
#     total_improvement = f1_df.iloc[-1] - initial_f1

#     # Calculate the incremental improvements and set negative increments to zero
#     incremental_improvements = f1_df.diff().fillna(f1_df.iloc[0]).clip(lower=0)

#     # Calculate percentage contributions for each class
#     percentage_contributions = (incremental_improvements.divide(total_improvement, axis=1) * 100).fillna(0)

#     print("Client Contributions to F1 Score Improvement:")
#     print(percentage_contributions)

#     # Print the final confusion matrix
#     print("Final Confusion Matrix:")
#     print(final_confusion_matrix)
#     print(type(final_confusion_matrix))
#     matrix_list = final_confusion_matrix.tolist()
#     return percentage_contributions, matrix_list



def calculate_reward_and_matrix():
    # Fetch paths and addresses
    dataset_paths, client_addresses = fetch_all_client_data_paths()
    if not dataset_paths:
        print("No datasets found. Exiting function.")
        return None, None  # Handle no data case

    # Prepare datasets only if file exists
    client_datasets = [prepare_dataset(pd.read_csv(path)) for path in dataset_paths if os.path.exists(path)]
    if not client_datasets:
        print("No client datasets prepared. Exiting function.")
        return None, None  # Handle no data case

    # Load and prepare the test dataset
    test_df = pd.read_csv('mnist-test/mnist_test.csv')
    test_dataset = prepare_dataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    incremental_dataset = None
    incremental_f1_scores = []

    # Process each client dataset
    for i, client_data in enumerate(client_datasets):
        if incremental_dataset is None:
            incremental_dataset = client_data
        else:
            incremental_dataset = ConcatDataset([incremental_dataset, client_data])

        train_loader = DataLoader(incremental_dataset, batch_size=64, shuffle=True)
        model = NeuralNetwork()

        # Evaluate the model
        accuracy, confusion, precision, recall, f1_score = train_and_evaluate(model, train_loader, test_loader)
        incremental_f1_scores.append(f1_score)
        print(f'Incremental accuracy after adding client {client_addresses[i]}: {accuracy:.2f}%')
        print("Metrics per class:")
        print(pd.DataFrame({
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        }, index=[f'Class {j}' for j in range(len(precision))]))
        print("\n")

    f1_df = pd.DataFrame(incremental_f1_scores, columns=[f'Class {i}' for i in range(10)])
    f1_df['Client'] = client_addresses
    f1_df.set_index('Client', inplace=True)

    initial_f1 = np.zeros(10)
    total_improvement = f1_df.iloc[-1] - initial_f1
    incremental_improvements = f1_df.diff().fillna(f1_df.iloc[0]).clip(lower=0)
    percentage_contributions = (incremental_improvements.divide(total_improvement.replace(0, np.nan), axis=1) * 100).fillna(0)

    # Sum and normalize contributions for each client
    summed_contributions = percentage_contributions.groupby('Client').sum()
    summed_contributions = summed_contributions.divide(summed_contributions.sum(), axis=1) * 100

    # Handle NaN values explicitly before conversion
    summed_contributions = summed_contributions.fillna(0).astype(int)

    print("Normalized Contributions to F1 Score Improvement by Client:")
    print(summed_contributions)

    print("Final Confusion Matrix:")
    print(confusion)
    return summed_contributions, confusion.tolist()

# calculate_reward_and_matrix()