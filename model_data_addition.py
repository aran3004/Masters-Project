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
from model_architecture import NeuralNetwork

# Initialize Web3
web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
file_storage_address = web3.to_checksum_address("0x49cbF6595A522AA113f9036Eb9f5D8Be666d6eAF")
contributions_address = web3.to_checksum_address("0x2301966a1C4eA1c8E138ceA8e33EC3727d5C9e09")

# Load the contract ABIs
with open('FileStorageABI.json', 'r') as file:
    file_storage_abi = json.load(file)
file_storage_contract = web3.eth.contract(address=file_storage_address, abi=file_storage_abi)

with open('ClientContributionsABI.json', 'r') as file:
    contributions_abi = json.load(file)
contributions_contract = web3.eth.contract(address=contributions_address, abi=contributions_abi)

# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28 * 28, 512),  # MNIST images are 28x28
#             nn.ReLU(),
#             nn.Linear(512, 10)  # 10 classes for MNIST digits
#         )

#     def forward(self, x):
#         x = self.flatten(x)
#         return self.linear_relu_stack(x)

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

    torch.save(model.state_dict(), 'saved_models/final_model.pth')

    return accuracy, cmatrix, precision, recall, f1_score


def fetch_all_client_data_paths():
    # Fetching all client data paths from the FileStorage contract
    client_addresses = file_storage_contract.functions.getAllClientAddresses().call()
    dataset_paths = []
    addresses = []
    for address in client_addresses:
        num_files = file_storage_contract.functions.getClientFileCount(address).call()
        for i in range(num_files):
            data_path = file_storage_contract.functions.getClientData(address, i).call()
            print(f'Data Path {i} : {data_path}')
            addresses.append(address)
            dataset_paths.append(data_path)
    return dataset_paths, addresses

def update_blockchain_contributions(address, contributions):
    """
    Updates or adds new client contributions to the blockchain.
    :param address: Ethereum address of the client
    :param contributions: Array of contributions to update
    """
    contributions_contract.functions.setContributions(address, contributions).transact({'from': web3.eth.default_account})


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
    final_accuracy = 0

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
        final_accuracy = accuracy
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

    # Assuming `summed_contributions` is the DataFrame from your reward and matrix function
    for client_address in summed_contributions.index:
        # Fetch the contribution values for the current client, converting each to integer
        contributions_array = [int(contribution) for contribution in summed_contributions.loc[client_address].values]
        tx_hash = contributions_contract.functions.setContributions(client_address, contributions_array).transact({'from': client_address})
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Optional: Print out the transaction receipt or some confirmation message
        print(f"Contributions updated for {client_address}. Transaction hash: {tx_hash.hex()}")


    print("Normalized Contributions to F1 Score Improvement by Client:")
    print(summed_contributions)

    print("Final Confusion Matrix:")
    print(confusion)
    accuracy_str = str(final_accuracy) + "%"
    print('Final Accuracy to be added to the Blockchain: ' + accuracy_str)
    tx_hash = file_storage_contract.functions.setAccuracy(accuracy_str).transact({'from': client_address})
    return summed_contributions, confusion.tolist()


# calculate_reward_and_matrix()