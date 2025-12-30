# Blackwell-MatMul
Optimized TF32 matrix multiplication kernel for RTX 50-series GPUs.

## Performance (RTX 5090)
Dimensions: 4096 x 4096 x 4096

| Clock             | Custom Kernel (TFLOPs) | cuBLAS 13.0 (TFLOPs) | Speedup    |
|-------------------|------------------------|----------------------|------------|     
| Unlimited         | **109.471**            | 103.645              |**+5.62%**  |
| Capped (2407 MHz) | **95.073**             | 87.151               |**+9.09%**  |

## Dependencies
* [Eigen](https://libeigen.gitlab.io)
* [CLI11](https://github.com/CLIUtils/CLI11)

## Build & Run
```
# Build
cmake . && make -j

# Run Benchmark
./matmul -b

# Run Correctness Verification (against Eigen CPU implementation)
./matmul -v
```

## Note
This is a reference implementation optimized for specific matrix dimensions. It prioritizes architectural clarity and peak performance demonstration over general-purpose flexibility, robustness, or edge-case handling.