from web3 import Web3
import json

def initilise_web3_contracts():
    web3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
    file_storage_address = web3.to_checksum_address("0xc789755244c38397B94A77D623A339887844AA83")
    contributions_address = web3.to_checksum_address("0xcc2FCAEEb892eC05A1bc05631382d8c0C62c0Be8")
    # Load the contract ABIs
    with open('FileStorageABI.json', 'r') as file:
        file_storage_abi = json.load(file)
    file_storage_contract = web3.eth.contract(address=file_storage_address, abi=file_storage_abi)

    with open('ClientContributionsABI.json', 'r') as file:
        contributions_abi = json.load(file)
    contributions_contract = web3.eth.contract(address=contributions_address, abi=contributions_abi)

    return web3, file_storage_contract, contributions_contract