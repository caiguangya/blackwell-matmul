/*
 * Copyright (c) 2025 Guangya Cai
 * 
 * Licensed under the MIT License. See LICENSE file in the project root for full license information.
 */

#include <cuda_runtime.h>

#include <cuda/barrier>

#include <iostream>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <random>

#include <Eigen/Dense>
#include <CLI/CLI.hpp>

template <int size> 
using BlockBarriers = cuda::barrier<cuda::thread_scope_block>[size];

using tfloat32 = float;

template <int size>
using MMAFragment = tfloat32[size];

template <int size>
using IntArray = int[size];

__forceinline__ __device__ void tf32conversion(float& r) {
    asm volatile ("cvt.rn.tf32.f32 %0, %1;\n"
        : "=f"(r) 
        : "f"(r)
    );
}

__forceinline__ __device__ void ldmatrix1(float& r0, const float* smemPtr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 { %0 }, [ %1 ];\n" 
        : "=f"(r0)
        : "l"(__cvta_generic_to_shared(smemPtr))
    );
}

__forceinline__ __device__ void ldmatrix2(float& r0, float& r1, const float* smemPtr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 { %0, %1 }, [ %2 ];\n" 
        : "=f"(r0), "=f"(r1)
        : "l"(__cvta_generic_to_shared(smemPtr))
    );
}

__forceinline__ __device__ void ldmatrix4(float& r0, float& r1, float& r2, float& r3, const float* smemPtr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n" 
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "l"(__cvta_generic_to_shared(smemPtr))
    );
}

__forceinline__ __device__ void stmatrix4(float& r0, float& r1, float& r2, float& r3, float* smemPtr) {
    asm volatile ("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :
        : "l"(__cvta_generic_to_shared(smemPtr)), "f"(r0), "f"(r1), "f"(r2), "f"(r3)
    );
}

__forceinline__ __device__ void mmaM16N8K8(float *d, const tfloat32 *a, const tfloat32 *b, const float *c) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]),
          "f"(b[0]), "f"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

__forceinline__ __device__ void mmaM16N8K4(float *d, const tfloat32 *a, const tfloat32 *b, const float *c) {
    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "f"(a[0]), "f"(a[1]), 
          "f"(b[0]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

float *gpuMalloc(std::size_t size) {
    float *mem = nullptr;

    auto err = cudaMalloc(&mem, size);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
        mem = nullptr;
    }

    return mem;
}

void gpuFree(float *mem) {
    auto err = cudaFree(mem);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

void copyToGPU(std::size_t size, const float *cpuVec, float *gpuVec) {
    auto err = cudaMemcpy(gpuVec, cpuVec, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << std::endl;
    }
}

template <int Row, int Col, CUtensorMapSwizzle swizzle, int tmaWidth> requires 
(swizzle == CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B || swizzle == CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B)
struct SharedMemHelper {
public:
    __forceinline__  __device__ int getOffset(int colIdx, int rowIdx) const noexcept {
        int actualCol = (rowIdx / tmaWidth) * Col + colIdx;
        int actualRow = rowIdx % tmaWidth;

        //XOR swizzling is just computing coset(s) of Z_2^n, what's the problem :)
        if constexpr (swizzle == CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B) {
            constexpr int mask = 4 - 1;

            return actualCol * tmaWidth + (actualRow ^ ((actualCol & mask) << 3));
        }
        else {
            constexpr int mask = 8 - 1;

            return actualCol * tmaWidth + (actualRow ^ ((actualCol & mask) << 2));
        }
    }
};

template <int threadCount, int smemBaseOffset, int smemAddlOffset, int mmaShapeM, int mmaShapeK, 
    int warpShapeM, class SharedMemHelper, int mmaFragSizeA>
__device__ void loadMMAFragmentA(const SharedMemHelper& helper,
           int bufferIdx, const float* source, int colIdxBase, int rowIdxBase, MMAFragment<mmaFragSizeA>& fragA) {
    constexpr int matrixCount = warpShapeM / mmaShapeM;
    constexpr int entryPerTheardTensor = (mmaShapeM * mmaShapeK) / threadCount;

    constexpr int minMMAShapeM = 8;
    constexpr int minMMAShapeK = 4;

    const float *pSourceBase = source + smemBaseOffset + bufferIdx * smemAddlOffset;

    int laneIdx = threadIdx.x % threadCount;
    int threadRowStartIdx = laneIdx / minMMAShapeK + rowIdxBase;
    int threadColStartIdx = laneIdx % minMMAShapeK + colIdxBase;

    int fragAIdx = 0;
#pragma unroll
    for (int t = 0; t < matrixCount; t++) {
#pragma unroll
        for (int j = 0; j < mmaShapeK; j += minMMAShapeK) {
            int threadColIdx = threadColStartIdx + j;
#pragma unroll
            for (int i = 0; i < mmaShapeM; i += minMMAShapeM) {
                int threadRowIdx = threadRowStartIdx + t * mmaShapeM + i;

                fragA[fragAIdx++] = pSourceBase[helper.getOffset(threadColIdx, threadRowIdx)];
            }
        }
    }
}

template <int mmaShapeN, int mmaShapeK, class SharedMemHelper, int fragmentSize> requires 
(mmaShapeN == 8 && (mmaShapeK == 8 || mmaShapeK == 4))
__forceinline__ __device__ void ldMatrix(const SharedMemHelper& helper, 
    int laneIdx, const float *pSourceBase, int colIdx, int rowIdx, int offset, MMAFragment<fragmentSize>& fragment) {
    constexpr int ldMatrixRowSize = 4;
    constexpr int ldMatrixColSize = 8;

    if constexpr (mmaShapeK == 8) {
        const int groupSize = 16;
        
        const float *ptr = pSourceBase + 
            helper.getOffset(colIdx + laneIdx % ldMatrixColSize + (laneIdx / groupSize) * mmaShapeN,
                rowIdx + ((laneIdx % groupSize) / ldMatrixColSize) * ldMatrixRowSize);
        ldmatrix4(fragment[offset], fragment[offset + 1], fragment[offset + 2], fragment[offset + 3], ptr);
    }
    else {
        const float *ptr = pSourceBase + helper.getOffset(colIdx + laneIdx, rowIdx);
        
        ldmatrix4(fragment[offset], fragment[offset + 1], fragment[offset + 2], fragment[offset + 3], ptr);
    }
}

template <int threadCount, int smemBaseOffset, int smemAddlOffset, int mmaShapeN, int mmaShapeK, 
    int warpShapeN, class SharedMemHelper, int mmaFragSizeB> requires (warpShapeN % 16 == 0)
__device__ void loadMMAFragmentB(const SharedMemHelper& helper,
           int bufferIdx, const float* source, int colIdxBase, int rowIdxBase, MMAFragment<mmaFragSizeB>& fragB) {
    constexpr int maxLdShapeN = 16;
    constexpr int entryPerTheardLdMatrix = 4;
    constexpr int ldMatrixCount = warpShapeN / maxLdShapeN;

    const float *pSource = source + smemBaseOffset + bufferIdx * smemAddlOffset;

    int laneIdx = threadIdx.x % threadCount;

#pragma unroll
    for (int t = 0; t < ldMatrixCount; t++) {
        ldMatrix<mmaShapeN, mmaShapeK>(helper, laneIdx, pSource,
            colIdxBase + t * maxLdShapeN, rowIdxBase, t * entryPerTheardLdMatrix, fragB);
    }
}

template <int tmaWidth, int smemBaseOffset, int smemAddlOffset, int smemShapeK, int smemShapeM, int barsTotalCount>
__forceinline__ __device__ void loadToSharedMemA(int offset, int bufferIdx, float* smemPtr, 
    const CUtensorMap& tensorMap, BlockBarriers<barsTotalCount>& barriers) {
    float *pSmemA = smemPtr + smemBaseOffset + smemAddlOffset * bufferIdx;

    constexpr int tmaRequestCountA = smemShapeM / tmaWidth;

    int rowIdx = blockIdx.x * smemShapeM;
    int colIdx = offset;

#pragma unroll
    for (int i = 0; i < tmaRequestCountA; i++) {
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(pSmemA + i * tmaWidth * smemShapeK, &tensorMap, 
            rowIdx + i * tmaWidth, colIdx, barriers[bufferIdx * tmaRequestCountA + i]);

        cuda::device::barrier_arrive_tx(barriers[bufferIdx * tmaRequestCountA + i], 1, smemShapeK * tmaWidth * sizeof(float));
    }
}

template <int tmaWidth, int smemBaseOffset, int smemAddlOffset, int smemShapeK, int smemShapeN, int barsTotalCount>
__forceinline__ __device__ void loadToSharedMemB(int offset, int bufferIdx, float* smemPtr, 
    const CUtensorMap& tensorMap, BlockBarriers<barsTotalCount>& barriers) {
    float *pSmemB = smemPtr + smemBaseOffset + smemAddlOffset * bufferIdx;

    constexpr int tmaRequestCountB = smemShapeK / tmaWidth;

    int colIdx = blockIdx.y * smemShapeN;
    int rowIdx = offset;

#pragma unroll
    for (int i = 0; i < tmaRequestCountB; i++) {
        cuda::device::experimental::cp_async_bulk_tensor_2d_global_to_shared(pSmemB + i * tmaWidth * smemShapeN, &tensorMap, 
            rowIdx + i * tmaWidth, colIdx, barriers[bufferIdx]);
    }

    cuda::device::barrier_arrive_tx(barriers[bufferIdx], 1, smemShapeK * smemShapeN * sizeof(float));
}

template <int mmaShapeM, int mmaShapeN, int mmaShapeK, int mmaFragASize, int mmaFragBSize, int mmaFragCSize> requires 
(mmaShapeM == 16) && (mmaShapeN == 8) && (mmaShapeK == 8 || mmaShapeK == 4)
__forceinline__ __device__ void mma(const MMAFragment<mmaFragASize>& fragA, int aOffset,
    const MMAFragment<mmaFragBSize>& fragB, int bOffset,
    MMAFragment<mmaFragCSize>& fragC, int cOffset) {

    if constexpr (mmaShapeK == 8) {
        mmaM16N8K8(&fragC[cOffset], &fragA[aOffset], &fragB[bOffset], &fragC[cOffset]);
    }
    else {
        mmaM16N8K4(&fragC[cOffset], &fragA[aOffset], &fragB[bOffset], &fragC[cOffset]);
    }
}

template <int threadCount, int warpShapeM, int warpShapeN, int mmaShapeM, int mmaShapeN, int mmaShapeK,
    int mmaFragASize, int mmaFragBSize, int mmaFragCSize>
__device__ void tensorMult(MMAFragment<mmaFragASize>& fragA, MMAFragment<mmaFragBSize>& fragB, 
    MMAFragment<mmaFragCSize>& fragC) {
    constexpr int tensorACount = warpShapeM / mmaShapeM;
    constexpr int entryPerTheardA = (mmaShapeM * mmaShapeK) / threadCount;

    constexpr int tensorBCount = warpShapeN / mmaShapeN;
    constexpr int entryPerTheardB = (mmaShapeN * mmaShapeK) / threadCount;

    constexpr int entryPerThreadC = (mmaShapeM * mmaShapeN) / threadCount;

    int cIdx = 0;
#pragma unroll
    for (int j = 0; j < tensorBCount; j++) {
#pragma unroll
        for (int i = 0; i < tensorACount; i++) {
            mma<mmaShapeM, mmaShapeN, mmaShapeK>(fragA, i * entryPerTheardA, fragB, j * entryPerTheardB, fragC, cIdx);
            cIdx += entryPerThreadC;
        }
    }
}

template <int threadCount, int warpShapeM, int warpShapeN, int mmaShapeM, int mmaShapeN, int mmaShapeK, int mmaFragCSize>
__device__ void writeToC(int colIdxBase, int rowIdxBase, MMAFragment<mmaFragCSize>& fragC, float *C, int ldc) {
    constexpr int entryPerThreadC = (mmaShapeM * mmaShapeN) / threadCount;
    static_assert(entryPerThreadC == 4);

    constexpr int colSubOffset = 1;
    constexpr int rowSubOffset = 8;

    int laneIdx = threadIdx.x % threadCount; 

    int threadLocalRowOffset =  laneIdx / 4;
    int threadLocalColOffset = (laneIdx % 4) * 2;

    int idx = 0;
#pragma unroll
    for (int j = 0; j < warpShapeN; j += mmaShapeN) {
        int colIdx = colIdxBase + j + threadLocalColOffset;
#pragma unroll
        for (int i = 0; i < warpShapeM; i += mmaShapeM) {
            int rowIdx = rowIdxBase + i + threadLocalRowOffset;

            C[colIdx * ldc + rowIdx] = fragC[idx + 0];
            C[colIdx * ldc + rowIdx + rowSubOffset] = fragC[idx + 2];
            C[(colIdx + colSubOffset) * ldc + rowIdx] = fragC[idx + 1];
            C[(colIdx + colSubOffset) * ldc + rowIdx + rowSubOffset] = fragC[idx + 3];

            idx += entryPerThreadC;
        }
    }
}

template <int barsTotalCount>
__forceinline__ __device__ void waitForBarrier(int idx, BlockBarriers<barsTotalCount>& bars, int phase) {
    while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bars[idx]), (phase & 1)));
}

template <int tmaRequestCount, int barsTotalCount, int barsPerWarpCount>
__forceinline__ __device__ void waitForBarriers(const IntArray<barsPerWarpCount>& barsIndices, 
    int bufferIdx, BlockBarriers<barsTotalCount>& bars, int phase) {
#pragma unroll
    for (int i = 0; i < barsPerWarpCount; i++) {
        while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bars[bufferIdx * tmaRequestCount + barsIndices[i]]), (phase & 1)));
    }
}

consteval bool kernelLaunchCheck(int ctaShapeM, int ctaShapeN, int smemShapeK, int warpShapeM, int warpShapeN,
          int mmaShapeM, int mmaShapeN, int mmaShapeK, int tmaWidth) {

    bool validCTAShape = (ctaShapeM % warpShapeM == 0) && (ctaShapeN % warpShapeN == 0);
    bool validWarpShape = (warpShapeM % mmaShapeM == 0) && (warpShapeN % mmaShapeN == 0);
    bool validSmemShape = (smemShapeK % mmaShapeK == 0) && (smemShapeK % tmaWidth == 0);

    return validCTAShape && validWarpShape && validSmemShape;
}

template <int ctaShapeM, int ctaShapeN, int smemShapeK, int warpShapeM, int warpShapeN,
          int mmaShapeM, int mmaShapeN, int mmaShapeK, int tmaWidth, int bufferCount, 
          int threadCount, std::uint64_t sharedMemAlignment> requires 
          (kernelLaunchCheck(ctaShapeM, ctaShapeN, smemShapeK, warpShapeM, warpShapeN, mmaShapeM, mmaShapeN, mmaShapeK, tmaWidth))
__global__ void matMul(const __grid_constant__ CUtensorMap tensorMapA, const __grid_constant__ CUtensorMap tensorMapB,
    int m, int n, int k,
    const float *A, int lda,
    const float *B, int ldb, 
    float *C, int ldc) {
    
    constexpr int consumerWarpCount = (ctaShapeM * ctaShapeN) / (warpShapeM *  warpShapeN);
    constexpr int consumerThreadsCount = consumerWarpCount * threadCount;

    int warpIdx = threadIdx.x / threadCount - 1;

    int warpRowIdx = (warpIdx % (ctaShapeM / warpShapeM)) * warpShapeM;
    int warpColIdx = (warpIdx / (ctaShapeM / warpShapeM)) * warpShapeN;
    
    int rowIdxBase = blockIdx.x * ctaShapeM;
    int colIdxBase = blockIdx.y * ctaShapeN;

    constexpr int mmaFragSizeA = (warpShapeM * mmaShapeK) / threadCount;
    constexpr int mmaFragSizeB = (warpShapeN * mmaShapeK) / threadCount;
    constexpr int mmaFragSizeC = (warpShapeM * warpShapeN) / threadCount;

    MMAFragment<mmaFragSizeA> fragA[2];
    MMAFragment<mmaFragSizeB> fragB[2];
    MMAFragment<mmaFragSizeC> fragC{};

    extern __shared__ float smem[];

    //Adjusting the alignment for TMA with swizzling working properly, it appears to use the pointer address to calculate the "column" for swizzling.
    float *smemPtr = smem + (sharedMemAlignment - ((std::uint64_t)smem & (sharedMemAlignment - 1))) / sizeof(float);

    constexpr CUtensorMapSwizzle swizzleA = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    constexpr CUtensorMapSwizzle swizzleB = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    constexpr SharedMemHelper<ctaShapeM, smemShapeK, swizzleA, tmaWidth> smemAHelper;
    constexpr SharedMemHelper<smemShapeK, ctaShapeN, swizzleB, tmaWidth> smemBHelper;

    constexpr int shareMemASize = ctaShapeM * smemShapeK;
    constexpr int shareMemBSize = ctaShapeN * smemShapeK;

    constexpr int smemAOffsets[2] = { 0, shareMemASize };
    constexpr int smemBOffsets[2] = { bufferCount * shareMemASize, shareMemBSize };
    
    constexpr int tmaRequestCountA = ctaShapeM / tmaWidth;

    __shared__ BlockBarriers<tmaRequestCountA * bufferCount> smemABarriers;
    __shared__ BlockBarriers<bufferCount> smemBBarriers;

    __shared__ BlockBarriers<bufferCount> bufferProcessedBarriers;

    constexpr int smemABarsPerWarpCount = warpShapeM / tmaWidth <= 0 ? 1 : warpShapeM / tmaWidth;

    if (threadIdx.x  < tmaRequestCountA * bufferCount) {
        init(smemABarriers + threadIdx.x, 1);
    }
    if (threadIdx.x < bufferCount) {
        init(smemBBarriers + threadIdx.x, 1);
        init(bufferProcessedBarriers + threadIdx.x, 1 + consumerThreadsCount);
    }
    if (threadIdx.x == 0) {
        cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    int pTileCount = (k - 1) / smemShapeK + 1;
    int pCurTile = 0;

    if (warpIdx < 0) {
        if (threadIdx.x == 0) {
#pragma unroll
            for (; pCurTile < bufferCount && pCurTile < pTileCount; pCurTile++) {
                loadToSharedMemA<tmaWidth, smemAOffsets[0], smemAOffsets[1], smemShapeK, ctaShapeM>(smemShapeK * pCurTile, 
                    pCurTile, smemPtr, tensorMapA, smemABarriers);
                loadToSharedMemB<tmaWidth, smemBOffsets[0], smemBOffsets[1], smemShapeK, ctaShapeN>(smemShapeK * pCurTile, 
                    pCurTile, smemPtr, tensorMapB, smemBBarriers);
            }

            for (; pCurTile < pTileCount; pCurTile++) {
                bufferProcessedBarriers[pCurTile % bufferCount].arrive_and_wait();

                loadToSharedMemA<tmaWidth, smemAOffsets[0], smemAOffsets[1], smemShapeK, ctaShapeM>(smemShapeK * pCurTile, 
                    pCurTile % bufferCount, smemPtr, tensorMapA, smemABarriers);
                loadToSharedMemB<tmaWidth, smemBOffsets[0], smemBOffsets[1], smemShapeK, ctaShapeN>(smemShapeK * pCurTile, 
                    pCurTile % bufferCount, smemPtr, tensorMapB, smemBBarriers);
            }
        }
    }
    else {
        IntArray<smemABarsPerWarpCount> smemABarsIndices;

#pragma unroll
        for (int i = 0; i < smemABarsPerWarpCount; i++) {
            smemABarsIndices[i] = warpRowIdx / tmaWidth + i;
        }

        waitForBarriers<tmaRequestCountA>(smemABarsIndices, 0, smemABarriers, 0);
        waitForBarrier(0, smemBBarriers, 0);
    
        loadMMAFragmentA<threadCount, smemAOffsets[0], smemAOffsets[1], mmaShapeM, mmaShapeK, warpShapeM>(smemAHelper, 0, smemPtr, 
            0, warpRowIdx, fragA[0]);
        loadMMAFragmentB<threadCount, smemBOffsets[0], smemBOffsets[1], mmaShapeN, mmaShapeK, warpShapeN>(smemBHelper, 0, smemPtr, 
            warpColIdx, 0, fragB[0]);

        int fragIdx = 0;

#pragma unroll 1
        while (pCurTile < pTileCount - 1) {
#pragma unroll
            for (int pLocal = 0; pLocal < smemShapeK; pLocal += mmaShapeK) {
                if (pLocal == smemShapeK - mmaShapeK) {
                    bufferProcessedBarriers[pCurTile % bufferCount].arrive();

                    pCurTile += 1;

                    waitForBarriers<tmaRequestCountA>(smemABarsIndices, pCurTile % bufferCount, smemABarriers, pCurTile / bufferCount);
                    waitForBarrier(pCurTile % bufferCount, smemBBarriers, pCurTile / bufferCount);
                }

                int prefetchIdx = (fragIdx + 1) % 2;
                int pPrefetch = (pLocal + mmaShapeK) % smemShapeK;
                
                loadMMAFragmentA<threadCount, smemAOffsets[0], smemAOffsets[1], mmaShapeM, mmaShapeK, warpShapeM>(
                    smemAHelper, pCurTile % bufferCount, smemPtr, pPrefetch, warpRowIdx, fragA[prefetchIdx]);
                loadMMAFragmentB<threadCount, smemBOffsets[0], smemBOffsets[1], mmaShapeN, mmaShapeK, warpShapeN>(
                    smemBHelper, pCurTile % bufferCount, smemPtr, warpColIdx, pPrefetch, fragB[prefetchIdx]);
                    
                tensorMult<threadCount, warpShapeM, warpShapeN, mmaShapeM, mmaShapeN, mmaShapeK>(fragA[fragIdx], fragB[fragIdx], fragC);

                fragIdx = prefetchIdx;
            }
        }
        {
#pragma unroll
            for (int pLocal = 0; pLocal < smemShapeK - mmaShapeK; pLocal += mmaShapeK) {  
                int prefetchIdx = (fragIdx + 1) % 2;
                int pPrefetch = pLocal + mmaShapeK;

                loadMMAFragmentA<threadCount, smemAOffsets[0], smemAOffsets[1], mmaShapeM, mmaShapeK, warpShapeM>(
                    smemAHelper, pCurTile % bufferCount, smemPtr, pPrefetch, warpRowIdx, fragA[prefetchIdx]);
                loadMMAFragmentB<threadCount, smemBOffsets[0], smemBOffsets[1], mmaShapeN, mmaShapeK, warpShapeN>(
                    smemBHelper, pCurTile % bufferCount, smemPtr, warpColIdx, pPrefetch, fragB[prefetchIdx]);
                    
                tensorMult<threadCount, warpShapeM, warpShapeN, mmaShapeM, mmaShapeN, mmaShapeK>(fragA[fragIdx], fragB[fragIdx], fragC);

                fragIdx = prefetchIdx;
            }

            tensorMult<threadCount, warpShapeM, warpShapeN, mmaShapeM, mmaShapeN, mmaShapeK>(fragA[fragIdx], fragB[fragIdx], fragC);
        }

        writeToC<threadCount, warpShapeM, warpShapeN, mmaShapeM, mmaShapeN, mmaShapeK>(colIdxBase + warpColIdx, rowIdxBase + warpRowIdx, 
            fragC, C, ldc);
    }
}

int main(int argc, char *argv[]) {
    bool verification = false;
    bool benchmark = false;

    CLI::App app;
    app.add_flag("-b", benchmark, "Benchmarking");
    app.add_flag("-v", verification, "Correctness verification");

    app.parse(argc, argv);

    int m = 4096;
    int n = 4096;
    int k = 4096;

    std::size_t sizeA = m * k * sizeof(float);
    std::size_t sizeB = k * n * sizeof(float);
    std::size_t sizeC = m * n * sizeof(float);

    std::srand(std::random_device{}());

    Eigen::MatrixXf A = Eigen::MatrixXf::Random(m, k);
    Eigen::MatrixXf B = Eigen::MatrixXf::Random(k, n);
    Eigen::MatrixXf C = Eigen::MatrixXf::Zero(m, n);

    float *gpuA = gpuMalloc(sizeA);
    if (gpuA == nullptr) return -1;

    float *gpuB = gpuMalloc(sizeB);
    if (gpuB == nullptr) return -1;

    float *gpuC = gpuMalloc(sizeC);
    if (gpuC == nullptr) return -1;

    copyToGPU(sizeA, A.data(), gpuA);
    copyToGPU(sizeB, B.data(), gpuB);

    constexpr int ctaShapeM = 64;
    constexpr int ctaShapeN = 64;

    constexpr int warpShapeM = 16;
    constexpr int warpShapeN = 32;

    constexpr int smemShapeK = 64;

    constexpr int tmaWidth = 32;
    constexpr int mmaShapeM = 16;
    constexpr int mmaShapeN = 8;
    constexpr int mmaShapeK = 8;

    constexpr int threadCount = 32;

    constexpr int bufferCount = 3;
    constexpr std::size_t shareMemASize = ctaShapeM * smemShapeK * sizeof(float);
    constexpr std::size_t shareMemBSize = ctaShapeN * smemShapeK * sizeof(float);

    //Slightly padding the allocated shared memory size for the alignment (see above)
    constexpr std::size_t sharedMemAlignment = 1024;
    constexpr std::size_t sharedMemSize = (shareMemASize + shareMemBSize) * bufferCount + sharedMemAlignment;

    dim3 grid((m + ctaShapeM - 1) / ctaShapeM, (n + ctaShapeN - 1) / ctaShapeN);
    dim3 block((ctaShapeM * ctaShapeN) / (warpShapeM * warpShapeN) * threadCount + threadCount);
    
    CUtensorMap tensorMapA{};
    CUtensorMap tensorMapB{};
    {
        constexpr int tensorRank = 2;
        std::uint64_t tensorSize[tensorRank] = {(std::uint64_t)m,  (std::uint64_t)k};
        std::uint64_t tensorStride[tensorRank - 1] = {m * sizeof(float)};
        std::uint32_t tensorBoxSize[tensorRank] =  {tmaWidth, smemShapeK};
        std::uint32_t tensorEleStride[tensorRank] = {1, 1};
        
        //Box inner dimension must be properly set for 128B swizzling to work properly, otherwise, it may be "swizzled" out of bound.
        auto ret = cuTensorMapEncodeTiled(&tensorMapA, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32, tensorRank, gpuA, 
            tensorSize, tensorStride, tensorBoxSize, tensorEleStride, 
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        if (ret != CUDA_SUCCESS) {
            std::cerr << "TMA initiation error, code: " << ret << std::endl;
        }
    }
    {
        constexpr int tensorRank = 2;
        std::uint64_t tensorSize[tensorRank] = {(std::uint64_t)k, (std::uint64_t)n};
        std::uint64_t tensorStride[tensorRank - 1] = {k * sizeof(float)};
        std::uint32_t tensorBoxSize[tensorRank] =  {tmaWidth, ctaShapeN};
        std::uint32_t tensorEleStride[tensorRank] = {1, 1};
        
        //Box inner dimension must be properly set for 128B swizzling to work properly, otherwise, it may be "swizzled" out of bound.
        auto ret = cuTensorMapEncodeTiled(&tensorMapB, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32, tensorRank, gpuB, 
            tensorSize, tensorStride, tensorBoxSize, tensorEleStride, 
            CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
            CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
            CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

        if (ret != CUDA_SUCCESS) {
            std::cerr << "TMA initiation error, code: " << ret << std::endl;
        }
    }

    cudaFuncSetAttribute(
        matMul<ctaShapeM, ctaShapeN, smemShapeK, warpShapeM, warpShapeN, mmaShapeM, mmaShapeN, mmaShapeK, tmaWidth, bufferCount, threadCount, sharedMemAlignment>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    
    matMul<ctaShapeM, ctaShapeN, smemShapeK, warpShapeM, warpShapeN, mmaShapeM, mmaShapeN, mmaShapeK, tmaWidth, bufferCount, threadCount, sharedMemAlignment><<<grid, block, sharedMemSize>>>(tensorMapA, tensorMapB, m, n, k, gpuA, m, gpuB, k, gpuC, m);

    {
        auto err = cudaGetLastError();
        
        if (err != cudaSuccess) {
            std::cerr << cudaGetErrorString(err) << std::endl;
            return -1;
        }
    }

    cudaMemcpy(C.data(), gpuC, sizeC, cudaMemcpyDeviceToHost);

    if (benchmark) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);

        constexpr int iterationCount = 10;
        for (int i = 0; i < iterationCount; i++) {
            matMul<ctaShapeM, ctaShapeN, smemShapeK, warpShapeM, warpShapeN, mmaShapeM, mmaShapeN, mmaShapeK, tmaWidth, bufferCount, threadCount, sharedMemAlignment><<<grid, block, sharedMemSize>>>(tensorMapA, tensorMapB, m, n, k, 
                gpuA, m, gpuB, k, gpuC, m);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedTime = 0.f;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        
        std::int64_t matMulFLO = (2LL * m) * (n * k);
        float tflops =  (matMulFLO * iterationCount * 1e-9) / elapsedTime;

        std::cout << "TFLOPS: " << tflops << std::endl;
    }

    if (verification) {
        if (C.isApprox(A * B, 1e-2)) {
            std::cout << "Passed the correctness verification" << std::endl;
        }
        else {
            std::cout << "Failed to pass the correctness verification" << std::endl;
        }
    }

    gpuFree(gpuA);
    gpuFree(gpuB);
    gpuFree(gpuC);

    return 0;
}