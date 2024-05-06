import torch.nn as nn
import torch
from model_architecture import NeuralNetwork
from model_data_addition import prepare_dataset
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from web3 import Web3
import json


# Connect to Ganache
ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Load contract ABIs
with open('FileStorageABI.json', 'r') as file_storage_abi_file:
    file_storage_abi = json.load(file_storage_abi_file)
with open('ClientContributionsABI.json', 'r') as contributions_abi_file:
    contributions_abi = json.load(contributions_abi_file)

# Contract addresses
file_storage_contract_address = web3.to_checksum_address("0xdc0e43e25ba2A572319eAAe6Bee21DB17DA16c73")
contributions_contract_address = web3.to_checksum_address("0xAE461517c0c6c05586B32f65F1238FcDD401421a")

# Contract instances
file_storage_contract = web3.eth.contract(address=file_storage_contract_address, abi=file_storage_abi)
contributions_contract = web3.eth.contract(address=contributions_contract_address, abi=contributions_abi)

def predictions(filename):

    model = NeuralNetwork()
    model.load_state_dict(torch.load('saved_models/final_model.pth'))
    model.eval()

    distribution_dataset = prepare_dataset(pd.read_csv(filename))
    distribution_loader = DataLoader(distribution_dataset, batch_size=64, shuffle=False)
    predictions = []
    contributions = []

    for inputs, labels in distribution_loader:
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

        # Print actual and predicted values
        for label, prediction in zip(labels, predicted):
            predictions.append(prediction.item())
    
    return predictions

def fetch_contributions():
    # Fetch all client addresses
    client_addresses = file_storage_contract.functions.getAllClientAddresses().call()
    
    # Dictionary to hold contributions data structured by class
    class_contributions = {i: {} for i in range(10)}  # Assuming 10 classes

    # Iterate over each client and fetch their contributions
    for address in client_addresses:
        contributions = contributions_contract.functions.getContributions(address).call()
        # Structure contributions by class
        for class_index in range(len(contributions)):
            if address not in class_contributions[class_index]:
                class_contributions[class_index][address] = contributions[class_index]
            else:
                class_contributions[class_index][address] += contributions[class_index]

    return class_contributions
