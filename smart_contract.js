*
* SPDX-License-Identifier: Apache-2.0
*/
'use strict';
const { Contract } = require('fabric-contract-api');
class EvidenceContract extends Contract {
async evidenceExists(ctx, evidenceId) {
const buffer = await ctx.stub.getState(evidenceId);
return (!!buffer && buffer.length > 0);
}
async createEvidence(ctx, evidenceId, value) {
const asset = { value };
const buffer = Buffer.from(JSON.stringify(asset));
await ctx.stub.putState(evidenceId, buffer);
return (`The record ${evidenceId} added`);
}
async readEvidence(ctx, evidenceId) {
const exists = await this.evidenceExists(ctx, evidenceId);
if(!exists) {
return (`The record ${evidenceId} does not exist`);
}
const buffer = await ctx.stub.getState(evidenceId);
TOC H INSTITUTE OF SCIENCE AND TECHNOLOGY
ARAKKUNNAM, ERNAKULAM, PIN – 682 313 Page No: 36
EVIDENCE MANAGEMENT USING BLOCKCHAIN
const asset = JSON.parse(buffer.toString());
return asset;
}
}
module.exports = EvidenceContract;
