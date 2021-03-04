# RNN-Drug-Generation
A Study on Recurrent Architecture for De Novo Drug Generation

In drug discovery, deep learning algorithms have emerged to become an effective method to generate novel chemical structures. They can speed up this process and decrease expenditure. We optimized the computational framework for \emph{de novo} drug design based on Recurrent Neural Networks that can learn the syntax of molecular representation in SMILES notation. We perform a comprehensive study on the architecture and hyper-parameters. Moreover, we compare two types of encoding and spatial arrangement of molecules: Embedding and One-hot Encoding and datasets with and without stereo-chemical information, respectively. The best model consists of an RNN containing 3 layers of Long Short-term Memory cells with 512 units each, a batch size of 16, the 'RMSProp' optimizer, and a sampling temperature of 0.75. We report improved results compared to the current literature regarding the validity and diversity of the generated SMILES. The best models reached values as high as $98.7\%$ valid generated SMILES for the ChEMBL datasets and $94.7\%$  for the ZINC biogenic library that contains stereo-chemical information. In both cases, the diversity of the generated compounds demonstrated the effectiveness of the recurrent architectures in learning the SMILES syntax and adding novelty to generate promising compounds. Note that the biogenetic dataset leads to even greater diversity, about $0.90$.

## Requirements
*  CUDA 10.1
*  NVIDIA GPU
*  Tensorflow 2.3
*  Python 3.8.3
*  Numpy
*  RDKit
*  tqdm
