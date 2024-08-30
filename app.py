from flask import Flask, render_template, request, redirect, make_response, session, url_for
from web3 import Web3
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import json
from model_data_addition import calculate_reward_and_matrix
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from distribution import predictions, fetch_contributions
from intialise_web3 import initilise_web3_contracts
matplotlib.use('Agg')  # Ensure this is set to prevent GUI issues



# Initialize Web3 Contracts
web3, file_storage_contract, contributions_contract = initilise_web3_contracts()

app = Flask(__name__)
app.jinja_env.globals.update(zip=zip)
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
    return render_template('models.html',  current_path=request.path)

@app.route('/about')
def about():
    return render_template('about.html', current_path=request.path)

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
    accuracy = file_storage_contract.functions.getAccuracy().call()
    number_contributors = len(client_addresses)
    print(f'Number of contributors: {len(client_addresses)}')
    return render_template('mnist.html', confusion_matrix=preprocessed_data, last_updated=last_updated_value, contributors = number_contributors, accuracy=accuracy)

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

@app.route('/query', methods=['POST', 'GET'])
def query():
    if request.method == 'POST':
        client_address = request.form['clientAddress']
        client_address = web3.to_checksum_address(client_address)
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join('temporary', filename)
            file.save(temp_path)
            return redirect(url_for('distribute', filename=temp_path, client_address=client_address))


    return render_template('query.html')

@app.route('/distribute', methods=['POST', 'GET'])
def distribute():
    filename = request.args.get('filename')
    client_address = request.args.get('client_address')

    # covering the case when page reloads
    if not os.path.exists(filename):
        return redirect('/models')
    
    df = pd.read_csv(filename)
    images = df.drop('label', axis=1).values

    encoded_images = []
    predicted = predictions(filename=filename)
    fetched_contributions = fetch_contributions()
    contributions = []  # List to store contributions info
    client_credits = {}  # Dictionary to store total credits due to each client

    for i in range(len(images)):
        image = images[i].reshape(28, 28)
        buf = BytesIO()
        plt.figure(figsize=(1, 1))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        # Encode the buffer to a base64 string
        base64_image = base64.b64encode(buf.getvalue()).decode('utf-8')
        encoded_images.append(base64_image)
        
        class_index = predicted[i]
        current_contributions = fetched_contributions.get(class_index, {})

        # Calculate credits for each client based on current contributions
        for client, percentage in current_contributions.items():
            actual_credit = 1 * (percentage / 100)  # Converting percentage to a fraction of 1 Ethereum
            if client not in client_credits:
                client_credits[client] = actual_credit  # Initialize if not already in dictionary
            else:
                client_credits[client] += actual_credit  # Add actual credit for this image

        contributions.append(current_contributions)
    
    os.remove(filename)

    total_credits = 0
    # Print out the credits due to each client
    for client, credits in client_credits.items():
        print(f"Client {client}: {credits} Ethereum")
        total_credits +=credits

    print(f'Total credits required for payment: {total_credits}')
    formatted_credits = "{:.2f}".format(total_credits)

    session['total_credits'] = total_credits
    session['client_credits'] = client_credits
    session['payee'] = client_address

    return render_template('distribute.html', images=encoded_images, predicted=predicted, contributions=contributions, total_credits=formatted_credits)


@app.route('/payment_distributed', methods=['POST'])
def payment_distributed():
    print('Starting payment')
    private_key = request.form['privateKey']
    payee = session.get('payee', '')
    print(f"payee: {payee}")
    total_credits = session.get('total_credits', 0)
    client_credits = session.get('client_credits', {})

    try:
        for client, credits in client_credits.items():
            # Ensure credits are in Wei if dealing with Ether
            credits_in_wei = web3.to_wei(credits, 'ether')

            # Transaction details
            nonce = web3.eth.get_transaction_count(payee)
            tx = {
                'nonce': nonce,
                'to': client,  # Client's account address
                'value': credits_in_wei,  # Amount of Ether to send
                'gas': 2000000,
                'gasPrice': web3.to_wei('50', 'gwei')
            }
            # Sign the transaction
            signed_tx = web3.eth.account.sign_transaction(tx, private_key)

            # Send the transaction
            tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            print(f"Transaction sent to {client}, hash: {tx_hash.hex()}")

            # Wait for the transaction to be mined (optional)
            tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"Transaction to {client} completed with receipt: {tx_receipt}")

    except Exception as e:
        print(f"An error occurred during payment distribution: {str(e)}")


    print(f'Total credits required for payment: {total_credits}')
    print('Client Credits:', client_credits)
    print('Payment would have been distributed')


    session.pop("payee", None)
    session.pop("total_credits", None)
    session.pop("client_credits", None)
    return redirect('/models')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
