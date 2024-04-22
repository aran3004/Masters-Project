// Using the window.onload event to make sure the entire page is loaded
window.onload = async function() {
    // Check if the MetaMask extension is installed in the browser
    if (window.ethereum) {
        // Initialize web3 instance using MetaMask's provider
        window.web3 = new Web3(ethereum);
        try {
            // Request account access using the modern eth_requestAccounts method
            const accounts = await ethereum.request({ method: 'eth_requestAccounts' });
            // Assuming the first account is the user's primary account
            const account = accounts[0];

            // Update the HTML content with the user's Ethereum address
            document.getElementById('account').innerText = account;

            // Fetch and display the account's Ethereum balance
            const balance = await web3.eth.getBalance(account);
            const balanceInEth = web3.utils.fromWei(balance, 'ether');
            document.getElementById('balance').innerText = balanceInEth + ' ETH';

            // Populate a hidden input field with the Ethereum address, if needed for form submission
            if (document.getElementById('clientAddress')) {
                document.getElementById('clientAddress').value = account;
            }

            // Listen for account changes and update the UI accordingly
            ethereum.on('accountsChanged', function (accounts) {
                const newAccount = accounts[0];
                document.getElementById('account').innerText = newAccount;
                document.getElementById('clientAddress').value = newAccount;
            });

            // Listen for chain changes and reload the page to maintain consistent data
            ethereum.on('chainChanged', chainId => {
                window.location.reload();
            });

            // Listen for disconnections and log them
            ethereum.on('disconnect', (error) => {
                console.log('MetaMask has been disconnected:', error);
                alert('MetaMask has been disconnected. Please reconnect.');
            });

        } catch (error) {
            // Handle errors, such as user denying account access
            console.error("User denied account access or an error occurred:", error);
        }
    } else {
        // Inform the user that MetaMask is not installed
        console.error("MetaMask is not installed. Please install MetaMask to interact with this site.");
    }
}
