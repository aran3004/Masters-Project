// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FileStorage {
    struct ClientData {
        uint256[10] contributions;
        string dataPath;
    }

    string public lastUpdated = "No current contributions";
    uint256[10][10] public confusionMatrix;

    mapping(address => ClientData[]) public clientContributions;
    address[] public clientAddresses; // Array to track all client addresses

    function setLastUpdated(string memory _newUpdate) public {
        lastUpdated = _newUpdate;
    }

    function getLastUpdated() public view returns (string memory) {
        return lastUpdated;
    }

    function initializeConfusionMatrix() public {
        for (uint256 i = 0; i < 10; i++) {
            for (uint256 j = 0; j < 10; j++) {
                confusionMatrix[i][j] = 0;
            }
        }
    }

    function getConfusionMatrix() public view returns (uint256[10][10] memory) {
        return confusionMatrix;
    }

    function getConfusionMatrixValue(
        uint256 row,
        uint256 col
    ) public view returns (uint256) {
        require(row < 10 && col < 10, "Index out of bounds");
        return confusionMatrix[row][col];
    }

    function updateConfusionMatrix(
        uint256 row,
        uint256 col,
        uint256 value
    ) public {
        require(row < 10 && col < 10, "Index out of bounds");
        confusionMatrix[row][col] = value;
    }

    function updateConfusionMatrixRow(
        uint256 rowIndex,
        uint256[10] calldata newRow
    ) public {
        require(rowIndex < 10, "Index out of bounds");
        for (uint256 j = 0; j < 10; j++) {
            confusionMatrix[rowIndex][j] = newRow[j];
        }
    }

    function addClientData(
        uint256[10] calldata _contributions,
        string calldata _dataPath
    ) public {
        ClientData memory newEntry = ClientData({
            contributions: _contributions,
            dataPath: _dataPath
        });
        clientContributions[msg.sender].push(newEntry);

        // Check if this is the first time the client is contributing
        if (clientContributions[msg.sender].length == 1) {
            clientAddresses.push(msg.sender); // Add new client address to the array
        }
    }

    function getClientData(
        address _client,
        uint index
    )
        public
        view
        returns (uint256[10] memory contributions, string memory dataPath)
    {
        require(
            index < clientContributions[_client].length,
            "Index out of bounds"
        );
        return (
            clientContributions[_client][index].contributions,
            clientContributions[_client][index].dataPath
        );
    }

    // Function to retrieve all client addresses
    function getAllClientAddresses() public view returns (address[] memory) {
        return clientAddresses;
    }

    function getAllContributionsForClient(
        address _client
    ) public view returns (ClientData[] memory) {
        return clientContributions[_client];
    }

    function updateClientContributions(
        address client,
        uint index,
        uint256[10] calldata newContributions
    ) public {
        require(
            index < clientContributions[client].length,
            "Index out of bounds"
        );
        clientContributions[client][index].contributions = newContributions;
    }
}
