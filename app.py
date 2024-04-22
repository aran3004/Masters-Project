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

# Set up the contract
# contract_address = web3.to_checksum_address("0xfb66B39B74a6e2059725efd4411780C90749636C")
contract_address = web3.to_checksum_address("0x88C3CE4777861c5C103E46A8B0849d1Ab1e27c11")

# Load contract ABI
with open('FileStorageABI.json', 'r') as abi_definition:
    contract_abi = json.load(abi_definition)



contract = web3.eth.contract(address=contract_address, abi=contract_abi)

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
    # Fetch the confusion matrix data from the smart contract
    confusion_matrix_data = []
    for i in range(10):
        row = []
        for j in range(10):
            row.append(contract.functions.getConfusionMatrixValue(i, j).call())
        confusion_matrix_data.append(row)
    preprocessed_data = [(i, row) for i, row in enumerate(confusion_matrix_data)]

    last_updated_value = contract.functions.getLastUpdated().call()
    client_addresses = contract.functions.getAllClientAddresses().call()
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

            # Send transaction to the blockchain
            tx_hash = contract.functions.addClientData(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                file_path
            ).transact({'from': client_address})

            tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

            # Update the last contribution time
            current_time = datetime.now().strftime('%Y-%m-%d%H:%M:%S')
            tx_hash_time = contract.functions.setLastUpdated(current_time).transact({'from': client_address})
            tx_receipt_time = web3.eth.wait_for_transaction_receipt(tx_hash_time)

            # Update confusion matrix
            percentage_contributions, matrix_list = calculate_reward_and_matrix()
            for i, row in enumerate(matrix_list):
                try:
                    tx_hash = contract.functions.updateConfusionMatrixRow(i, row).transact({'from': client_address})
                    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
                    print(f"Row {i} updated on blockchain; Transaction hash: {receipt.transactionHash.hex()}")
                except Exception as e:
                    print("Error sending transaction:", e)
            
            # Update the contributions on the blockchain
            contributions_array = [int(contribution) for contribution in percentage_contributions.loc[client_address].values]
            update_index = 0

            tx_hash = contract.functions.updateClientContributions(client_address, update_index, contributions_array).transact({'from': client_address})
            tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"Contributions updated on blockchain; Transaction hash: {tx_receipt.transactionHash.hex()}")


        return redirect('/models')
    return render_template('contribute.html')


if __name__ == '__main__':
    app.run(debug=True)