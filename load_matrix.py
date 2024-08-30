from web3 import Web3
import json

# Connect to Ganache
ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Set up the contract
contract_address = web3.to_checksum_address("0x45Fe902959AAbDefA4B518CC17bc7CC67CF114F4")

# Load contract ABI
with open('FileStorageABI.json', 'r') as abi_definition:
    contract_abi = json.load(abi_definition)

# Create the contract instance with the ABI and Address
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

def fetch_confusion_matrix():
    # Get the confusion matrix from the smart contract
    matrix = contract.functions.getConfusionMatrix().call()
    print("Confusion Matrix:")
    for row in matrix:
        print(row)

# Execute the function
fetch_confusion_matrix()
