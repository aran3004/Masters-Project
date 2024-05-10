from web3 import Web3
import json

def initilise_web3_contracts():
    web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
    file_storage_address = web3.to_checksum_address("0xF5b53A389c2D6661C606C05943C7B05F0a9D929e")
    contributions_address = web3.to_checksum_address("0x3F87A3e6eEe19980552FFA8e3a3356B5618e66A3")
    # Load the contract ABIs
    with open('FileStorageABI.json', 'r') as file:
        file_storage_abi = json.load(file)
    file_storage_contract = web3.eth.contract(address=file_storage_address, abi=file_storage_abi)

    with open('ClientContributionsABI.json', 'r') as file:
        contributions_abi = json.load(file)
    contributions_contract = web3.eth.contract(address=contributions_address, abi=contributions_abi)

    return web3, file_storage_contract, contributions_contract