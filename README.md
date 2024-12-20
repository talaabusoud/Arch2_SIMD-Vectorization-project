# SIMD Optimization for Matrix and Vector Operations

## Introduction
This project demonstrates the application of SIMD (Single Instruction, Multiple Data) instructions using SSE (Streaming SIMD Extensions) to optimize numerical algorithms, specifically matrix-vector and matrix-matrix multiplications. By comparing scalar implementations with SIMD-enhanced versions, we show how parallel processing can significantly improve performance for these operations. This report details the implementation of vectorized algorithms using GCC and SSE intrinsics and evaluates the performance improvements achieved through SIMD optimization.

## Device Specifications
- **CPU Generation**: 11th Gen Intel® Core™ i7-1165G7
- **RAM Size**: 16.00 GB
- **L1 Cache**: 320 KB
- **L2 Cache**: 5.0 MB
- **L3 Cache**: 12.0 MB
- **Operating System**: Ubuntu 20.04.6
- **Environment**: Native Linux

## Results
The SIMD vectorization results indicate consistent performance improvements across different operations and input sizes. Below is a summary of the observed performance improvements:

- **Vector-Vector Multiplication**: The vectorized approach consistently demonstrates a 3% improvement over the scalar method.
- **Matrix-Vector Multiplication**: The vectorized implementation showcases notable improvements of approximately 2.34%, with larger input sizes yielding higher gains.
- **Matrix-Matrix Multiplication**: The vectorized approach consistently outperforms the scalar method, with percentage improvements ranging from approximately 1.81% to 2.18%.

These results demonstrate the effectiveness of SIMD vectorization in enhancing performance across various computational tasks and its potential for optimizing mathematical operations in parallel processing environments.

## Conclusion
SIMD vectorization has proven to be a highly effective optimization technique for computational tasks, including vector-vector, matrix-vector, and matrix-matrix multiplications. The results show consistent performance improvements of approximately 1.8% to 3% compared to scalar implementations. These gains highlight SIMD's advantages in parallel processing environments, demonstrating its significant impact on overall computational performance and its value in enhancing the speed and efficiency of high-performance computing tasks.

## Files in the Repository
- `MatrixMatrix.cpp`: Contains the implementation for matrix-matrix multiplication.
- `MatrixVector.cpp`: Contains the implementation for matrix-vector multiplication.
- `simd.cpp`: Contains the SIMD-enhanced vector operations using SSE intrinsics.
  
## Dependencies
- GCC or Clang compiler with support for SSE intrinsics.
- A system with an Intel or compatible CPU with SSE support.



