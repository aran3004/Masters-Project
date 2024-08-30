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

    // Function to transfer Ether from this contract to a specified recipient
    function transferEther(address payable recipient, uint256 amount) public {
        require(
            address(this).balance >= amount,
            "Insufficient balance to transfer"
        );
        // Perform the transfer
        (bool success, ) = recipient.call{value: amount}("");
        require(success, "Failed to send Ether");
    }

    // Fallback function to accept Ether when sent to the contract
    receive() external payable {}

    // Function to allow manual sending of Ether to the contract
    function depositEther() public payable {}

    // Function to check the contract's balance
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}
