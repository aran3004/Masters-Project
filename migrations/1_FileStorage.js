const FileStorage = artifacts.require("FileStorage");

module.exports = function(deployer){
    deployer.deploy(FileStorage).then(function(instance) {
        return instance.initializeConfusionMatrix();
      });
};