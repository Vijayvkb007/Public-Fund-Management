const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PublicFundTreasury", function () {
    let Treasury, treasury, owner, addr1, addr2, addr3;

    // Deploy contract before each test
    beforeEach(async function () {
        [owner, addr1, addr2, addr3] = await ethers.getSigners();

        Treasury = await ethers.getContractFactory("PublicFundTreasury");
        treasury = await Treasury.deploy(); // Fixed: Removed .deployed()
    });

    describe("Fund Management", function () {
        it("Should allow deposits and update fund balance", async function () {
            await treasury.connect(addr1).depositFunds({ value: ethers.parseEther("1") });

            const balance = await treasury.getTreasuryBalance();
            expect(balance).to.equal(ethers.parseEther("1"));
        });

        it("Should reject zero-value deposits", async function () {
            await expect(treasury.connect(addr1).depositFunds({ value: 0 })).to.be.revertedWith(
                "Deposit must be greater than zero"
            );
        });
    });

    describe("Proposal Management", function () {
        it("Should allow submitting a proposal", async function () {
            await treasury.connect(addr1).submitProposal("Build a park", ethers.parseEther("0.5"), addr2.address);

            const proposal = await treasury.getProposal(1);
            expect(proposal.description).to.equal("Build a park");
            expect(proposal.amount).to.equal(ethers.parseEther("0.5"));
            expect(proposal.recipient).to.equal(addr2.address);
            expect(proposal.votes).to.equal(0);
        });

        it("Should reject invalid proposal amounts", async function () {
            await expect(
                treasury.connect(addr1).submitProposal("Invalid proposal", 0, addr2.address)
            ).to.be.revertedWith("Requested amount must be greater than zero");
        });

        it("Should reject invalid recipient address", async function () {
            await expect(
                treasury.connect(addr1).submitProposal("No recipient", ethers.parseEther("1"), ethers.ZeroAddress)
            ).to.be.revertedWith("Invalid recipient address");
        });
    });

    describe("Voting", function () {
        beforeEach(async function () {
            await treasury.connect(addr1).submitProposal("Build a library", ethers.parseEther("1"), addr2.address);
        });

        it("Should allow voting on a proposal", async function () {
            await treasury.connect(addr1).voteOnProposal(1);
            const proposal = await treasury.getProposal(1);
            expect(proposal.votes).to.equal(1);
        });

        it("Should prevent double voting on the same proposal", async function () {
            await treasury.connect(addr1).voteOnProposal(1);
            await expect(treasury.connect(addr1).voteOnProposal(1)).to.be.revertedWith("Already voted on this proposal");
        });

        it("Should allow the same user to vote on different proposals", async function () {
            await treasury.connect(addr1).submitProposal("Build a school", ethers.parseEther("0.5"), addr3.address);
            await treasury.connect(addr1).voteOnProposal(1);
            await treasury.connect(addr1).voteOnProposal(2);

            const proposal1 = await treasury.getProposal(1);
            const proposal2 = await treasury.getProposal(2);
            expect(proposal1.votes).to.equal(1);
            expect(proposal2.votes).to.equal(1);
        });

        it("Should approve a proposal after reaching vote threshold", async function () {
            await treasury.connect(addr1).voteOnProposal(1);
            await treasury.connect(addr2).voteOnProposal(1);
            await treasury.connect(addr3).voteOnProposal(1);

            const proposal = await treasury.getProposal(1);
            expect(proposal.approved).to.equal(true);
        });

        it("Should reject voting on non-existent proposals", async function () {
            await expect(treasury.connect(addr1).voteOnProposal(999)).to.be.revertedWith("Proposal does not exist");
        });
    });

    describe("Fund Release", function () {
        beforeEach(async function () {
            await treasury.connect(owner).depositFunds({ value: ethers.parseEther("5") });
            await treasury.connect(addr1).submitProposal("Build a hospital", ethers.parseEther("2"), addr2.address);
        });

        it("Should release funds only for approved proposals", async function () {
            await treasury.connect(addr1).voteOnProposal(1);
            await treasury.connect(addr2).voteOnProposal(1);
            await treasury.connect(addr3).voteOnProposal(1);

            await expect(treasury.connect(owner).releaseFunds(1)).to.changeEtherBalances(
                [addr2, treasury],
                [ethers.parseEther("2"), ethers.parseEther("-2")]
            );
        });

        it("Should prevent fund release for non-approved proposals", async function () {
            await expect(treasury.connect(owner).releaseFunds(1)).to.be.revertedWith("Proposal is not approved");
        });

        it("Should prevent releasing funds more than once", async function () {
            await treasury.connect(addr1).voteOnProposal(1);
            await treasury.connect(addr2).voteOnProposal(1);
            await treasury.connect(addr3).voteOnProposal(1);

            await treasury.connect(owner).releaseFunds(1);
            await expect(treasury.connect(owner).releaseFunds(1)).to.be.revertedWith("Funds already released");
        });

        it("Should prevent releasing funds for non-existent proposals", async function () {
            await expect(treasury.connect(owner).releaseFunds(999)).to.be.revertedWith("Proposal does not exist");
        });
    });

    describe("Treasury Balance", function () {
        it("Should show the correct treasury balance", async function () {
            await treasury.connect(addr1).depositFunds({ value: ethers.parseEther("3") });
            const balance = await treasury.getTreasuryBalance();
            expect(balance).to.equal(ethers.parseEther("3"));
        });
    });
});
