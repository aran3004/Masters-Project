# Masters Project - Semi-Decentralised Collaborative AI Platform

This project aims to combine collaborative AI ideas and Ethereum Blockchain to create a 'stepping stone' platform towards a fully decentralised one. The platform utilises the Flask framework to communicate between the Blockchain, smart contract and front-end.

# Project Structure
build: Auto-generated when soldity files are migrated to the blockchain
contracts: Contain the solidity smart contract files for updating confusion matrices, contribution distributions, model metrics and payment methods
datasets: Local storage of files until FL can replace this part
migrations: JS files to facilitate migration of contracts to the blockchain
mnist_test.csv: Example of each digit for payment distribution demonstration
saved_models: All models will be saved here
static/connectWallet.js: Connecting MetaMask wallet from browser to the platform
static/displayInfo.js: Displaying the account details associated with the connected account
static/style.css: contains all the stylings used for the front-end
templates: Contains all the HTML files for the front-end. These are used by Flask routing
app.py: main code file responsible for Flask routing, rendering and communicating with other python files and blockchain
base_model.py: used to initially test process of training a model and communicating changes to the blockchain
ClientContribtuionsABI.json, FileStorageABI.json: Parts of the autogenreated files when migrating that are required to use the blockchain in Python
distribution.py: Responisble for all distribution changes and retrieval of information from the blockchain
initialise_web3.py: Speaks to the ...ABI.json files and initialises those contracts as seen in the local blockchain.
load_matrix.py:  Retrieval of confusion matrix information
model_architecture.py: Neural Network setup and parameters
model_data_addition.py: All model training and model performance comparisions which are then used to identify improvements and then payment distribution changes
testing_data_blockchain.py: Fetching all client data from blockchain
testing.py:Fetching all client data from blockchain


# Installation and Running the app
This project requires a local version of the Ethereum Blockchain running on the local device using Truffle Ganache. Once this has been setup, all the contracts need to be migrated to the blockchain using the Truffle Console and the "truffle migrate" command. The new contract addresses' should be updated in the Python files.

Use the 'requirements.txt' file to install all the dependencies

To run the app, run "python app.py" in the terminal
