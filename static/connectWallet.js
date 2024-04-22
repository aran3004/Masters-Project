document.getElementById("connectMetaMask").addEventListener("click", async () => {
    if (window.ethereum) { // Check if MetaMask is installed
        console.log("MetaMask Installed")
    //   try {
    //     // Will Start the MetaMask extension
    //     const accounts = await window.ethereum.request({ method: "eth_requestAccounts" });
    //     console.log("Connected", accounts[0]); // The address of the first account
    //   } catch (error) {
    //     console.error("Error connecting to MetaMask", error);
    //   }
    } else {
      alert("Please install MetaMask to use this feature!");
    }
  });