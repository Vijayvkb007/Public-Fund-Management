const { ethers } = require("hardhat");
const fs = require("fs");
require("dotenv").config();

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log("Deploying contracts with the account:", deployer.address);

  // Deploy PublicKeyRegistry contract
  console.log("Deploying PublicKeyRegistry...");
  const PublicKeyRegistry = await ethers.getContractFactory("PublicFundTreasury");
  const publicKeyRegistry = await PublicKeyRegistry.deploy(2);
  await publicKeyRegistry.waitForDeployment();
  const publicKeyRegistryAddress = await publicKeyRegistry.getAddress();
  console.log("PublicKeyRegistry deployed to:", publicKeyRegistryAddress);

  // Update .env file
  const envContent = `NEXT_PUBLIC_PUBLIC_FUND_TREASURY_ADDRESS=${publicKeyRegistryAddress}\n`;

  fs.writeFileSync("/home/sagar0418/0418/SKP-PFM/frontend/.env", envContent);

  console.log(".env file updated successfully!");

  console.log("\nDeployment Summary:");
  console.log("-------------------");
  console.log("PublicKeyRegistry:", publicKeyRegistryAddress);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
