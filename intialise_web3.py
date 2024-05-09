from web3 import Web3
import json

def initilise_web3_contracts():
    web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
    file_storage_address = web3.to_checksum_address("0x972D27E1ddc0e1aB4b038F8989A67f5FEae8f061")
    contributions_address = web3.to_checksum_address("0x93a2201C2Ea912ee0Cfc2aaDe03b1843016e9EEF")
    # Load the contract ABIs
    with open('FileStorageABI.json', 'r') as file:
        file_storage_abi = json.load(file)
    file_storage_contract = web3.eth.contract(address=file_storage_address, abi=file_storage_abi)

    with open('ClientContributionsABI.json', 'r') as file:
        contributions_abi = json.load(file)
    contributions_contract = web3.eth.contract(address=contributions_address, abi=contributions_abi)

    return web3, file_storage_contract, contributions_contract