---
title: "Accelerated Computing: GPUs"
date: 2026-01-06
draft: True
description: "On things I learned about GPUs"
tags: ["Technical blog"]
---
Last fall, I took a class [6.s894 Accelerated Computing](https://accelerated-computing.academy/fall25/) at MIT, and I cannot recommend it more. It teaches you all about GPUs and writing fast code for them.


# Matrix Multiplication
A workload we studied in depth was matrix multiplication. It is a very important problem that is the core of ML models today. Libraries like cuBLAS have highly optimized GEMM kernels, but it's interesting to take the problem apart and learn about what optimizations really make it fast.
The simplest implementation of a GEMM (general matrix multiplication) kernel doesn't assume anything about sparsity or structure and can be described by the following code.

```python
# input matrices: a (I × K), b (K × J)
# output matrix: c (I × J)
for i in range(I):
    for j in range(J):
        s = 0
        for k in range(K):
            s += a[i, k] * b[k, j]
        c[i, j] = s
```

## GEMM on the RTX 4000 Ada
Techniques for a fast gemm kernel involve strategies like exploiting tiling and reuse, improving SM occupancy with pipelining, and using tensor cores.

The first lab of the sequence teaches exploitation of tiling and reuse because simply getting all operands from L2 or DRAM makes a very slow kernel. Exploiting L1 shared memory is much faster, and registers even faster.

1. L1 Reuse
Work becomes partitions
each block computes a tile of the output matrix,

and values from the input matrices are brought into fast, on-chip memory (shared memory or L1 cache) and reused many times — reducing costly global memory traffic.

The lab encourages an output-stationary dataflow, where each thread block focuses on computing one tile of output while streaming through corresponding tiles of input. You also think about how to get the L1 to actually cache loads you want reused — for example by using read-only loads (__ldg) or explicit shared memory.

2. Register-Level Reuse (Part 3)

Once you’ve gotten a decent L1-aware GEMM going, the lab challenges you to push further: target a much faster runtime of < 8 ms by reusing data inside registers. Because registers are the fastest memory on the GPU, the idea here is to decompose the tile into microtiles that fit in registers and keep the inner loop’s operands there instead of reloading from L1.

This part also introduces practical performance engineering considerations like register pressure and how the compiler allocates registers, because spilling out of registers can drastically hurt performance.

## GEMM on the H100

