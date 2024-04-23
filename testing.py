from web3 import Web3
import json

# Connect to Ganache
ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Set up the contract for File Storage
file_storage_address = web3.to_checksum_address("0x9FDdA4a90A70536A4e871f255BA0679184eDa58B")  # Replace with actual address
with open('FileStorageABI.json', 'r') as file:
    file_storage_abi = json.load(file)
file_storage_contract = web3.eth.contract(address=file_storage_address, abi=file_storage_abi)

# Set up the contract for Client Contributions
contributions_address = web3.to_checksum_address("0x0215Cf2a37dFAd5d8651E4834431d0ba8FEd66EB")  # Replace with actual address
with open('ClientContributionsABI.json', 'r') as file:
    contributions_abi = json.load(file)
contributions_contract = web3.eth.contract(address=contributions_address, abi=contributions_abi)

def fetch_all_client_data():
    # Get all client addresses from the File Storage contract
    client_addresses = file_storage_contract.functions.getAllClientAddresses().call()
    print(f'Number of contributors: {len(client_addresses)}')
    if len(client_addresses) == 0:
        print("There are currently no contributors.")
    else:
        # Iterate over each address and fetch their file path and contributions
        for address in client_addresses:
            # Fetch file path
            # data_path = file_storage_contract.functions.getClientData(address).call()
            # print(f"Address: {address}")
            # print(f"  Data Path: {data_path}")
            
            # Fetch contributions
            contributions = contributions_contract.functions.getContributions(address).call()
            print(f"  Contributions from {address} : {contributions}")

# Execute the function
fetch_all_client_data()
