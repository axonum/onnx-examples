// SPDX-License-Identifier: MIT
pragma solidity >=0.4.16 <=0.8.23;

import "./AILib.sol";

contract LRContract {
    bytes constant MODEL = hex"080312077079746f7263681a05322e312e313acd090a2f0a05696e707574120e2f436173745f6f75747075745f301a052f436173742204436173742a090a02746f1801a001020a87010a0e2f436173745f6f75747075745f300a0d6c696e6561722e7765696768740a0b6c696e6561722e6269617312152f6c696e6561722f47656d6d5f6f75747075745f301a0c2f6c696e6561722f47656d6d220447656d6d2a0f0a05616c706861150000803fa001012a0e0a0462657461150000803fa001012a0d0a067472616e73421801a001020a3d0a152f6c696e6561722f47656d6d5f6f75747075745f3012112f5369676d6f69645f6f75747075745f301a082f5369676d6f696422075369676d6f69640a3f12122f436f6e7374616e745f6f75747075745f301a092f436f6e7374616e742208436f6e7374616e742a140a0576616c75652a0810014a040000003fa001040a440a112f5369676d6f69645f6f75747075745f300a122f436f6e7374616e745f6f75747075745f30120e2f4c6573735f6f75747075745f301a052f4c65737322044c6573730a2a0a0e2f4c6573735f6f75747075745f30120d2f4e6f745f6f75747075745f301a042f4e6f7422034e6f740a4312142f436f6e7374616e745f315f6f75747075745f301a0b2f436f6e7374616e745f312208436f6e7374616e742a140a0576616c75652a0810014a040000003fa001040a4a0a112f5369676d6f69645f6f75747075745f300a142f436f6e7374616e745f315f6f75747075745f3012102f4c6573735f315f6f75747075745f301a072f4c6573735f3122044c6573730a3e0a102f4c6573735f315f6f75747075745f3012102f436173745f315f6f75747075745f301a072f436173745f312204436173742a090a02746f1807a001020a4712142f436f6e7374616e745f325f6f75747075745f301a0b2f436f6e7374616e745f322208436f6e7374616e742a180a0576616c75652a0c100b4a080000000000000000a001040a420a142f436f6e7374616e745f325f6f75747075745f3012102f436173745f325f6f75747075745f301a072f436173745f322204436173742a090a02746f1807a001020a3e0a102f436173745f315f6f75747075745f300a102f436173745f325f6f75747075745f30120d2f4d756c5f6f75747075745f301a042f4d756c22034d756c0a3b0a0d2f4e6f745f6f75747075745f3012102f436173745f335f6f75747075745f301a072f436173745f332204436173742a090a02746f1807a001020a3b0a102f436173745f335f6f75747075745f300a0d2f4d756c5f6f75747075745f30120d2f4164645f6f75747075745f301a042f41646422034164640a310a0d2f4164645f6f75747075745f3012066f75747075741a072f436173745f342204436173742a090a02746f1807a00102120a6d61696e5f67726170682a3f0801080a1001420d6c696e6561722e7765696768744a281e92773eab62863e12b997bd29bb943e26e68dbdc4af823deda61dbe1a2c3e3eb9ba8e3eba8f6dbe2a1708011001420b6c696e6561722e626961734a04f9ba8c3e5a170a05696e707574120e0a0c080712080a0208010a02080a5a1f0a0d6c696e6561722e776569676874120e0a0c080112080a0208010a02080a5a190a0b6c696e6561722e62696173120a0a08080112040a02080162180a066f7574707574120e0a0c080712080a0208010a02080142021007";

    event LogisticRegression(bytes inputBytes, int64 output);

    function logisticRegression(int64[10] memory input) public returns (int64) {
        bytes memory inputBytes = encodeArray(input);
        bytes memory outputBytes = AILib.onnxInference(MODEL, inputBytes, 8);
        int64 output;
        assembly {
            output := mload(add(outputBytes, 8))
        }
        emit LogisticRegression(inputBytes, output);
        return output;
    }

    function encodeArray(int64[10] memory arr) internal pure returns (bytes memory) {
        bytes memory packed;
        for (uint256 i = 0; i < 10; i++) {
            packed = abi.encodePacked(packed, arr[i]);
        }
        return packed;
    }
}