// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FileStorage {
    string public fileName;
    string public lastUpdated; // Store date and time as a string
    uint256 public versionCounter = 0; // Initialize the version counter

    function setfileName(
        string memory _fileName,
        string memory _dateTime
    ) public {
        fileName = _fileName;
        lastUpdated = _dateTime;
        versionCounter += 1;
    }

    function getfileName() public view returns (string memory) {
        return fileName;
    }

    function getLastUpdated() public view returns (string memory) {
        return lastUpdated;
    }

    // Function to get the current version counter
    function getVersionCounter() public view returns (uint256) {
        return versionCounter;
    }
}
