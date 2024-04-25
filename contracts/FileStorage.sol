// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FileStorage {
    struct ClientData {
        string dataPath;
    }

    string public lastUpdated = "No current contributions";
    uint256[10][10] public confusionMatrix;
    mapping(address => string[]) public clientFiles;
    address[] public clientAddresses; // Array to track all client addresses
    string public accuracy = "No current contributions so 0 accuracy";

    function setAccuracy(string memory _accuracy) public {
        // You can add access control modifiers here
        accuracy = _accuracy;
    }

    function getAccuracy() public view returns (string memory) {
        return accuracy;
    }

    function addClientData(
        address clientAddress,
        string calldata _dataPath
    ) public {
        // Security checks needed here to ensure only authorized users can call this function
        clientFiles[clientAddress].push(_dataPath);
        if (clientFiles[clientAddress].length == 1) {
            clientAddresses.push(clientAddress); // Add new client address to the array
        }
    }

    function getClientData(
        address _client,
        uint index
    ) public view returns (string memory dataPath) {
        return clientFiles[_client][index];
    }

    function getClientFileCount(address client) public view returns (uint) {
        return clientFiles[client].length;
    }

    function getAllClientAddresses() public view returns (address[] memory) {
        return clientAddresses;
    }

    // Function to update the last updated timestamp
    function setLastUpdated(string memory _newUpdate) public {
        lastUpdated = _newUpdate;
    }

    // Function to get the last updated timestamp
    function getLastUpdated() public view returns (string memory) {
        return lastUpdated;
    }

    // Initialize the confusion matrix with zeros
    function initializeConfusionMatrix() public {
        for (uint i = 0; i < 10; i++) {
            for (uint j = 0; j < 10; j++) {
                confusionMatrix[i][j] = 0;
            }
        }
    }

    // Get a specific value from the confusion matrix
    function getConfusionMatrixValue(
        uint row,
        uint col
    ) public view returns (uint256) {
        require(row < 10 && col < 10, "Index out of bounds");
        return confusionMatrix[row][col];
    }

    // Update an entire row in the confusion matrix
    function updateConfusionMatrixRow(
        uint rowIndex,
        uint256[10] calldata newRow
    ) public {
        require(rowIndex < 10, "Index out of bounds");
        for (uint j = 0; j < 10; j++) {
            confusionMatrix[rowIndex][j] = newRow[j];
        }
    }
}
