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
from intialise_web3 import initilise_web3_contracts

# Initialize Web3 Contracts
web3, file_storage_contract, contributions_contract = initilise_web3_contracts()


# Set seeds for reproducibility
torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

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


def train_and_evaluate_heterogenous(model, train_loader, test_loader, epochs=5, final=False):
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
    precision, recall, _, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0)
    accuracy = np.trace(cmatrix) / np.sum(cmatrix)

    if final:
        torch.save(model.state_dict(), 'saved_models/final_model.pth')

    return accuracy * 100, cmatrix, precision, recall

def normalize_credits(row):
    total = row.sum()
    if total == 0:
        # If the sum is zero, return the row as is, or set an equal distribution if needed
        return row  # or return pd.Series([100/len(row)]*len(row), index=row.index)
    else:
        return (row / total) * 100

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

    # ------------------- Checking overlap ---------------------
    class_lists = []

    # Load class data for each client
    for file in dataset_paths:
        df = pd.read_csv(file)
        class_lists.append(set(df.iloc[:, 0]))  # first column is the class label
    
    # Create a matrix to check for shared classes between clients
    overlap_matrix = pd.DataFrame(index=range(1, len(dataset_paths)+1), columns=range(1, len(dataset_paths)+1), data=0)

    for i in range(len(dataset_paths)):
        for j in range(len(dataset_paths)):
            # If there's intersection between class sets of two different clients, mark as overlap
            if i != j and class_lists[i].intersection(class_lists[j]):
                overlap_matrix.iloc[i, j] = 1
    
    # Sum of each row will indicate how many other clients share at least one class with this client
    overlap_sums = overlap_matrix.sum(axis=1)

    # Check if there's any client with zero overlaps
    no_overlap_clients = overlap_sums[overlap_sums == 0].count()

    print(f"Overlap matrix:")
    print(overlap_matrix)
    print(f"Number of clients with no overlap in Config 1: {no_overlap_clients}")
    print(f"The current data is {'completely heterogeneous' if no_overlap_clients == len(dataset_paths) else 'not completely heterogeneous'}")

    # ----------------------------------------------------------

    # Prepare datasets only if file exists
    client_datasets = [prepare_dataset(pd.read_csv(path)) for path in dataset_paths if os.path.exists(path)]
    if not client_datasets:
        print("No client datasets prepared. Exiting function.")
        return None, None  # Handle no data case
    # Load and prepare the test dataset
    test_df = pd.read_csv('mnist-test/mnist_test.csv')
    test_dataset = prepare_dataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
    if no_overlap_clients == len(dataset_paths):
        # put code for heterogenous model here
        full_dataset = ConcatDataset(client_datasets)
        full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)
        model = NeuralNetwork()
        full_accuracy, full_cmatrix, full_precision, full_recall = train_and_evaluate_heterogenous(model, full_loader, test_loader, final=True)
        contributions_df = pd.DataFrame(index=[f'Class {i}' for i in range(10)], columns=client_addresses)

        if len(client_datasets) == 1:
            # Evaluate the single client dataset
            train_loader = DataLoader(client_datasets[0], batch_size=64, shuffle=True)
            model = NeuralNetwork()
            accuracy, cmatrix, precision, recall, f1_score = train_and_evaluate(model, train_loader, test_loader)
            contributions = [100 if f > 0 else 0 for f in f1_score]  # 100% contribution to classes with F1 score > 0

            contributions_df.iloc[:, 0] = contributions

        else:
            for excluded_client in range(len(client_datasets)):
                reduced_data = [data for i, data in enumerate(client_datasets) if i != excluded_client]
                reduced_loader = DataLoader(ConcatDataset(reduced_data), batch_size=64, shuffle=True)
                model = NeuralNetwork()
                reduced_accuracy, reduced_cmatrix, reduced_precision, reduced_recall = train_and_evaluate_heterogenous(model, reduced_loader, test_loader)

                print(f'Accuracy after removing client {client_addresses[excluded_client]}: {reduced_accuracy:.2f}%')
                full_diag = full_cmatrix.diagonal()
                reduced_diag = reduced_cmatrix.diagonal()

                # Safely calculate contributions
                contribution = np.zeros_like(full_diag)
                for i in range(len(full_diag)):
                    if full_diag[i] != 0:
                        contribution[i] = (full_diag[i] - reduced_diag[i]) / full_diag[i]
                        # Ensure no negative contributions
                        if contribution[i] < 0:
                            contribution[i] = 0
                    else:
                        contribution[i] = 0  

                contributions_df.iloc[:, excluded_client] = contribution * 100  # Convert to percentage

        # Normalize contributions to distribute 100 credits per class
        normalized_credits = contributions_df.apply(normalize_credits, axis=1)

        print("Normalized Credit Distribution:")
        transpose_df = normalized_credits.T
        transpose_df.index.name = 'Client'
        print(transpose_df)

        # Sum and normalize contributions for each client
        summed_contributions = transpose_df.groupby('Client').sum()
        summed_contributions = summed_contributions.divide(summed_contributions.sum(), axis=1) * 100

        # Handle NaN values explicitly before conversion
        summed_contributions = summed_contributions.fillna(0).astype(int)
        print(summed_contributions)

        # Blockchain interaction
        for client_address in summed_contributions.index:
            # Fetch the contribution values for the current client, converting each to integer
            contributions_array = [int(contribution) for contribution in summed_contributions.loc[client_address].values]
            tx_hash = contributions_contract.functions.setContributions(client_address, contributions_array).transact({'from': client_address})
            tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"Contributions updated for {client_address}. Transaction hash: {tx_hash.hex()}")


        print("Final Confusion Matrix:")
        print(full_cmatrix)
        accuracy_str = str(full_accuracy) + "%"
        print('Final Accuracy to be added to the Blockchain: ' + accuracy_str)
        tx_hash = file_storage_contract.functions.setAccuracy(accuracy_str).transact({'from': client_address})

        return summed_contributions, full_cmatrix.tolist()
        
    else:
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
        print(percentage_contributions)

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