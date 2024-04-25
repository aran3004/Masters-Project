import torch.nn as nn
import torch
from model_architecture import NeuralNetwork
from model_data_addition import prepare_dataset
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

model = NeuralNetwork()
model.load_state_dict(torch.load('saved_models/final_model.pth'))
model.eval()


distribution_dataset = prepare_dataset(pd.read_csv('mnist_for_distribution.csv'))
distribution_loader = DataLoader(distribution_dataset, batch_size=64, shuffle=False)



# Loop through the data_loader
for inputs, labels in distribution_loader:
    # Assume inputs and labels are torch tensors
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability

    # Print actual and predicted values
    for label, prediction in zip(labels, predicted):
        print(f'Actual label: {label.item()} - Predicted label: {prediction.item()}')

display_dataset = pd.read_csv('mnist_for_distribution.csv')

labels = display_dataset.iloc[:, 0].values  # Extract labels if present
images = display_dataset.iloc[:, 1:].values  # Extract image data

# Reshape the image data and plot the first few images
for i in range(10):  # Adjust the range to display more or fewer images
    image = images[i].reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {labels[i]}')
    plt.show()