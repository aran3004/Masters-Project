import torch.nn as nn
import torch
from model_architecture import NeuralNetwork
from model_data_addition import prepare_dataset
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from web3 import Web3
import json
from intialise_web3 import initilise_web3_contracts

web3, file_storage_contract, contributions_contract = initilise_web3_contracts()

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
    print(class_contributions)
    return class_contributions
