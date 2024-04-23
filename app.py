from flask import Flask, render_template, request, redirect, make_response, session
from web3 import Web3
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json
from model_data_addition import calculate_reward_and_matrix

# Connect to Ganache
ganache_url = "HTTP://127.0.0.1:7545"
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Load contract ABIs
with open('FileStorageABI.json', 'r') as file_storage_abi_file:
    file_storage_abi = json.load(file_storage_abi_file)
with open('ClientContributionsABI.json', 'r') as contributions_abi_file:
    contributions_abi = json.load(contributions_abi_file)

# Contract addresses
file_storage_contract_address = web3.to_checksum_address("0x9FDdA4a90A70536A4e871f255BA0679184eDa58B")
contributions_contract_address = web3.to_checksum_address("0x0215Cf2a37dFAd5d8651E4834431d0ba8FEd66EB")

# Contract instances
file_storage_contract = web3.eth.contract(address=file_storage_contract_address, abi=file_storage_abi)
contributions_contract = web3.eth.contract(address=contributions_contract_address, abi=contributions_abi)

app = Flask(__name__)
app.secret_key = 'my_masters_project_code'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/models')
def models():
    return render_template('models.html')

@app.route('/mnist')
def mnist():
    # Fetch the confusion matrix data from the file storage smart contract
    confusion_matrix_data = []
    for i in range(10):
        row = []
        for j in range(10):
            row.append(file_storage_contract.functions.getConfusionMatrixValue(i, j).call())
        confusion_matrix_data.append(row)
    preprocessed_data = [(i, row) for i, row in enumerate(confusion_matrix_data)]

    last_updated_value = file_storage_contract.functions.getLastUpdated().call()
    client_addresses = file_storage_contract.functions.getAllClientAddresses().call()
    number_contributors = len(client_addresses)
    print(f'Number of contributors: {len(client_addresses)}')
    return render_template('mnist.html', confusion_matrix=preprocessed_data, last_updated=last_updated_value, contributors = number_contributors)

@app.route('/contribute', methods=['POST', 'GET'])
def contribute():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            new_filename = f"{filename.split('.')[0]}_{str(datetime.now())}.csv"
            file.save(os.path.join("datasets", new_filename))
            file_path = os.path.join("datasets", new_filename)
            client_address = request.form['clientAddress']
            client_address = web3.to_checksum_address(client_address)

            # Send file path to the file storage contract
            tx_hash = file_storage_contract.functions.addClientData(client_address, file_path).transact({'from': client_address})
            web3.eth.wait_for_transaction_receipt(tx_hash)

            # Update the last contribution time using the file storage contract
            current_time = datetime.now().strftime('%Y-%m-%d%H:%M:%S')
            tx_hash_time = file_storage_contract.functions.setLastUpdated(current_time).transact({'from': client_address})
            web3.eth.wait_for_transaction_receipt(tx_hash_time)


            # Update confusion matrix
            percentage_contributions, matrix_list = calculate_reward_and_matrix()
            for i, row in enumerate(matrix_list):
                try:
                    tx_hash = file_storage_contract.functions.updateConfusionMatrixRow(i, row).transact({'from': client_address})
                    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"Row {i} updated on blockchain; Transaction hash: {receipt.transactionHash.hex()}")
                except Exception as e:
                    print("Error sending transaction:", e)

        return redirect('/models')
    return render_template('contribute.html')

if __name__ == '__main__':
    app.run(debug=True)
