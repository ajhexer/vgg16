
# VGG16 Neural Network Implementation on CPU and GPU

This repository contains an implementation of the VGG16 neural network architecture, developed in two stages: first on the CPU and then on the GPU. This project builds upon an another project that provided initial implementations for both [CPU](https://github.com/tigercosmos/simple-vgg16) and [GPU](https://github.com/tigercosmos/simple-vgg16-cu/tree/master). The CPU part was implemented using OpenMP, and the GPU part utilized NVIDIA’s cuBLAS and cuDNN libraries.

## Project Overview

### CPU Implementation
- **Original Implementation**: The CPU version of VGG16 was originally developed using OpenMP to parallelize operations, enabling multi-core processing.
- **Optimization and Analysis**: I conducted a thorough performance analysis to identify bottlenecks in the original CPU code, which is documented in the included PowerPoint presentation. Based on these findings, I made several optimizations to improve the execution efficiency.

### GPU Implementation
- **Original Implementation**: The GPU version leveraged NVIDIA’s cuBLAS and cuDNN libraries for accelerated computation.
- **Re-Implementation**: I re-implemented key functions from cuBLAS from scratch, tailored specifically for the VGG16 architecture, to optimize performance and reduce reliance on third-party libraries.

## Results and Analysis
The PowerPoint presentation includes detailed analysis and visualizations of the CPU implementation's performance improvements. The GPU implementation is compared against the original cuBLAS-based approach, highlighting the efficiency gains from the re-implementation.
