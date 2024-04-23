const ClientContributions = artifacts.require("ClientContributions");

module.exports = function (deployer) {
    deployer.deploy(ClientContributions);
};
