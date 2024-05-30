// SPDX-License-Identifier: MIT
pragma solidity >=0.4.16 <=0.8.23;

import "./AILib.sol";

contract ReluContract {
    bytes constant MODEL = hex"080312077079746f7263681a05322e312e313ad3010a2f0a05696e707574120e2f436173745f6f75747075745f301a052f436173742204436173742a090a02746f1801a001020a2d0a0e2f436173745f6f75747075745f30120e2f52656c755f6f75747075745f301a052f52656c75220452656c750a320a0e2f52656c755f6f75747075745f3012066f75747075741a072f436173745f312204436173742a090a02746f1807a00102120a6d61696e5f67726170685a170a05696e707574120e0a0c080712080a0208010a02080262180a066f7574707574120e0a0c080712080a0208010a02080242021007";

    uint256 constant INT64_SIZE = 8;

    event ReluInference(int64 input1, int64 input2, int64 output1, int64 output2);

    function reluInference(int64 input1, int64 input2) public returns (int64 output1, int64 output2) {
        bytes memory input = abi.encodePacked(input1, input2);
        bytes memory output = AILib.onnxInference(MODEL, input, 16);
        (output1, output2) = unpack(output);
        emit ReluInference(input1, input2, output1, output2);
        return (output1, output2);
    }

    function unpack(bytes memory output) internal pure returns (int64, int64) {
        int64 output1;
        int64 output2;
        assembly {
            output1 := mload(add(output, INT64_SIZE))
            output2 := mload(add(output, mul(INT64_SIZE, 2)))
        }
        return (output1, output2);
    }
}