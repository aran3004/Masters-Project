// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ClientContributions {
    // Mapping from client address to their contributions (array of 10 integers)
    mapping(address => uint256[10]) public contributions;

    // Event for logging updates to contributions
    event ContributionsUpdated(
        address indexed client,
        uint256[10] newContributions
    );

    // Function to update or set the contributions for a client
    function setContributions(
        address client,
        uint256[10] calldata newContributions
    ) public {
        contributions[client] = newContributions;
        emit ContributionsUpdated(client, newContributions);
    }

    // Function to retrieve contributions for a client
    function getContributions(
        address client
    ) public view returns (uint256[10] memory) {
        return contributions[client];
    }
}
