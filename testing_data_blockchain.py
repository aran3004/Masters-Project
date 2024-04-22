from web3 import Web3
import json

# Connect to Ganache
ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Set up the contract
# contract_address = web3.to_checksum_address("0xfb66B39B74a6e2059725efd4411780C90749636C")
contract_address = web3.to_checksum_address("0x88C3CE4777861c5C103E46A8B0849d1Ab1e27c11")

# Load contract ABI
with open('FileStorageABI.json', 'r') as abi_definition:
    contract_abi = json.load(abi_definition)

# Create the contract instance with the ABI and Address
contract = web3.eth.contract(address=contract_address, abi=contract_abi)


def fetch_all_client_data():
    # Get all client addresses
    client_addresses = contract.functions.getAllClientAddresses().call()
    print(f'Number of contributors: {len(client_addresses)}')
    if len(client_addresses) == 0:
        print("there are currently no contributions")
    else:
        # Iterate over each address and fetch their contributions
        for address in client_addresses:
            contributions = contract.functions.getAllContributionsForClient(address).call()
            print(f"Contributions for {address}:")
            print(contributions[0])
            # for data in contributions:
            #     contributions_list = data[0]  # Accessing first element of the tuple for contributions. array of integers
            #     data_path = data[1]          # Accessing second element of the tuple for data path. string
            #     print(f"  Contributions: {contributions_list}")
            #     print(f"  Data Path: {data_path}")
   

# Execute the function
fetch_all_client_data()
