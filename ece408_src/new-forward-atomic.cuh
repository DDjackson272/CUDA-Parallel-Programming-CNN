#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define TILE_WIDTH 16
#define CUDA_MAX_NUM_THREADS 512

namespace mxnet {
    namespace op {

        __constant__ float constK[15000];

//        __global__ void forward_kernel(float *__restrict__ y,
//                                       const float *__restrict__ x, const int M, const int C, const int H,
//                                       const int W, const int K, const int W_grid) {
//
//            const int H_out = H - K + 1;
//            const int W_out = W - K + 1;
//            const int h = ((blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y) * 2;
//            const int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
//            const int h2 = h + 1;
//            //const int w2 = w + TILE_WIDTH;
//
//            const bool flag1 = h < H_out && w < W_out;
//            //const bool flag2 = h < H_out && w2 < W_out;
//            const bool flag3 = h2 < H_out && w < W_out;
//            //const bool flag4 = h2 < H_out && w2 < W_out;
//
//            float acc1 = 0.0;
//            //float acc2 = 0.0;
//            float acc3 = 0.0;
//            //float acc4 = 0.0;
//
//            const int b = blockIdx.x;
//            const int m = blockIdx.y;
//
//            int iter = K / 2;
//            if (iter * 2 < K)
//                iter += 1;
//
//#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//#pragma unroll
//            for (int c = 0; c < C; ++c) {
//                for (int p = 0; p < iter; ++p) {
//                    for (int q = 0; q < iter; ++q) {
//                        int p1 = 2 * p;
//                        int p2 = 2 * p + 1;
//
//                        int q1 = 2 * q;
//                        int q2 = 2 * q + 1;
//
//                        if (flag1) acc1 += x4d(b, c, h + p1, w + q1) * k4d(m, c, p1, q1);
//                        //if (flag2) acc2 += x4d(b, c, h + p1, w2 + q1) * k4d(m, c, p1, q1);
//                        if (flag3) acc3 += x4d(b, c, h2 + p1, w + q1) * k4d(m, c, p1, q1);
//                        //if (flag4) acc4 += x4d(b, c, h2 + p1, w2 + q1) * k4d(m, c, p1, q1);
//
//                        if (q2 < K && p2 < K) {
//                            if (flag1)
//                                acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2) +
//                                        x4d(b, c, h + p2, w + q2) * k4d(m, c, p2, q2) +
//                                        x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
////                            if (flag2)
////                                acc2 += x4d(b, c, h + p1, w2 + q2) * k4d(m, c, p1, q2) +
////                                        x4d(b, c, h + p2, w2 + q2) * k4d(m, c, p2, q2) +
////                                        x4d(b, c, h + p2, w2 + q1) * k4d(m, c, p2, q1);
//                            if (flag3)
//                                acc3 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2) +
//                                        x4d(b, c, h2 + p2, w + q2) * k4d(m, c, p2, q2) +
//                                        x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
////                            if (flag4)
////                                acc4 += x4d(b, c, h2 + p1, w2 + q2) * k4d(m, c, p1, q2) +
////                                        x4d(b, c, h2 + p2, w2 + q2) * k4d(m, c, p2, q2) +
////                                        x4d(b, c, h2 + p2, w2 + q1) * k4d(m, c, p2, q1);
//                        } else if (q2 < K) {
//                            if (flag1) acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2);
//                            //if (flag2) acc2 += x4d(b, c, h + p1, w2 + q2) * k4d(m, c, p1, q2);
//                            if (flag3) acc3 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2);
//                            //if (flag4) acc4 += x4d(b, c, h2 + p1, w2 + q2) * k4d(m, c, p1, q2);
//                        } else if (p2 < K) {
//                            if (flag1) acc1 += x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
//                            //if (flag2) acc2 += x4d(b, c, h + p2, w2 + q1) * k4d(m, c, p2, q1);
//                            if (flag3) acc3 += x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
//                            //if (flag4) acc4 += x4d(b, c, h2 + p2, w2 + q1) * k4d(m, c, p2, q1);
//                        }
//                    }
//                }
//            }
//
//            if (flag1) y4d(b, m, h, w) = acc1;
//            //if (flag2) y4d(b, m, h, w2) = acc2;
//            if (flag3) y4d(b, m, h2, w) = acc3;
//            //if (flag4) y4d(b, m, h2, w2) = acc4;
//
//#undef y4d
//#undef x4d
//#undef k4d
//        }
//
        __global__ void forward_kernel_atomicAdd(float *__restrict__ y, const float *__restrict__ x,
                                                 const int B, const int M, const int C, const int H,
                                                 const int W, const int K, const int W_grid) {

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            const int h = ((blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y);
            const int w = ((blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x);

            const bool flag1 = h < H_out && w < W_out;

            int iter = K / 2;
            if (iter * 2 < K)
                iter += 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            int b = blockIdx.x;
            int m = blockIdx.y % M;
            int c = blockIdx.y / M;

            float acc1 = 0.0;

#pragma unroll

            for (int p = 0; p < iter; ++p) {
                for (int q = 0; q < iter; ++q) {
                    int p1 = 2 * p;
                    int p2 = 2 * p + 1;

                    int q1 = 2 * q;
                    int q2 = 2 * q + 1;

                    if (flag1) acc1 += x4d(b, c, h + p1, w + q1) * k4d(m, c, p1, q1);

                    if (q2 < K && p2 < K) {
                        if (flag1)
                            acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h + p2, w + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                    } else if (q2 < K) {
                        if (flag1) acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2);
                    } else if (p2 < K) {
                        if (flag1) acc1 += x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                    }
                }
            }

            if (flag1) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h * W_out + w], acc1);

#undef y4d
#undef x4d
#undef k4d
        }

//        __global__ void forward_kernel_optimized_shared_memory_convolution_with_loop_unrolling_with_atomic
//                (float *__restrict__ y, const float *__restrict__ x, const int M,
//                 const int C, const int H, const int W, const int K, const int W_grid,
//                 const int sm_H, const int sm_W) {
//
//            const int H_out = H - K + 1;
//            const int W_out = W - K + 1;
//            extern __shared__ float shmem[];
//
//            float *X_shared = &shmem[0];
//
//            const int b = blockIdx.x;
//            const int m = blockIdx.y;
//            const int h = (blockIdx.z / W_grid) * TILE_WIDTH * 2 + threadIdx.y;
//            const int h2 = h  + TILE_WIDTH;
//            const int w = (blockIdx.z % W_grid) * TILE_WIDTH * 2 + threadIdx.x;
//            const int w2 = w + TILE_WIDTH;
//
//            const bool flag1 = h < H_out && w < W_out;
//            const bool flag2 = h2 < H_out && w < W_out;
//            const bool flag3 = h < H_out && w2 < W_out;
//            const bool flag4 = h2 < H_out && w2 < W_out;
//
//            float acc  = 0.0;
//            float acc2 = 0.0;
//            float acc3 = 0.0;
//            float acc4 = 0.0;
//
//#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//#define x2d(i1, i0)  X_shared[i1 * sm_W + i0]
//#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//#pragma unroll
//
//            for (int c = 0; c < C; ++c){
//                for (int i = 0; i < sm_H - threadIdx.y; i += TILE_WIDTH) {
//                    for (int j = 0; j < sm_W - threadIdx.x; j += TILE_WIDTH) {
//                        x2d ((threadIdx.y + i), (threadIdx.x + j)) = x4d(b, c, (h + i), (w + j));
//                    }
//                }
//                __syncthreads();
//
//#pragma unroll
//                for (int p = 0; p < K; ++p) {
//                    for (int q = 0; q < K; ++q) {
//                        float mask = k4d(m, c, p, q);
//                        acc  += x2d ((threadIdx.y + p), (threadIdx.x + q)) * mask;
//                        acc2 += x2d ((threadIdx.y + TILE_WIDTH + p), (threadIdx.x + q)) * mask;
//                        acc3 += x2d ((threadIdx.y + p), (threadIdx.x + TILE_WIDTH + q)) * mask;
//                        acc4 += x2d ((threadIdx.y + TILE_WIDTH + p), (threadIdx.x + TILE_WIDTH + q)) * mask;
//                    }
//                }
//                __syncthreads();
//            }
//
//            if (flag4) {
//                y4d(b, m, h, w) = acc;
//                y4d(b, m, h2, w)= acc2;
//                y4d(b, m, h, w2)= acc3;
//                y4d(b, m, h2, w2)= acc4;
//            } else if (flag3) {
//                y4d(b, m, h, w)= acc;
//                y4d(b, m, h, w2)= acc3;
//            } else if (flag2) {
//                y4d(b, m, h, w) = acc;
//                y4d(b, m, h2, w)= acc2;
//            } else if (flag1) {
//                y4d(b, m, h, w) = acc;
//            }
//#undef y4d
//#undef x4d
//#undef x2d
//#undef k4d
//        }

//        __global__ void forward_kernel_optimized_shared_memory_convolution
//                (float *k, float *y, const float *x, const int M,
//                 const int C, const int H, const int W, const int K, const int W_grid) {
//            const int H_out = H - K + 1;
//            const int W_out = W - K + 1;
//
//            int X_tile_width = TILE_WIDTH + K - 1;
//            extern __shared__ float shmem[];
//
//            float *X_shared = &shmem[0];
//            float *K_shared = X_shared + X_tile_width * X_tile_width;
//
//            int b = blockIdx.x;
//            int m = blockIdx.y;
//
//            int h0 = threadIdx.y;
//            int w0 = threadIdx.x;
//
//            int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
//            int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;
//
//            int h = h_base + h0;
//            int w = w_base + w0;
//
//            float acc = 0.0;
//
//#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//            for (int c = 0; c < C; c++) {
//
//                if (h0 < K && w0 < K) {
//                    K_shared[h0 * K + w0] = k4d(m, c, h0, w0);
//                }
//                __syncthreads();
//
//                for (int i = h0; i < X_tile_width; i += TILE_WIDTH) {
//                    for (int j = w0; j < X_tile_width; j += TILE_WIDTH) {
//                        X_shared[i * X_tile_width + j] = x4d(b, c, i + h_base, j + w_base);
//                    }
//                }
//                __syncthreads();
//
//                for (int p = 0; p < K; ++p) {
//                    for (int q = 0; q < K; ++q) {
//                        int pos_x = (h0 + p) * X_tile_width + (w0 + q);
//                        int pos_k = p * K + q;
//                        acc += X_shared[pos_x] * K_shared[pos_k];
//                    }
//                }
//                __syncthreads();
//            }
//
//            if (h < H_out && w < W_out) {
//                y4d(b, m, h, w) = acc;
//            }
//
//#undef y4d
//#undef x4d
//#undef k4d
//        }


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called,
   so here we specialize with only floats.
*/
        template<>
        void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                                 const mshadow::Tensor<gpu, 4, float> &x,
                                 const mshadow::Tensor<gpu, 4, float> &k) {

            // Extract the tensor dimensions into B,M,C,H,W,K
            const int B = x.shape_[0];
            const int M = y.shape_[1];
            const int C = x.shape_[1];
            const int H = x.shape_[2];
            const int W = x.shape_[3];
            const int K = k.shape_[3];

            // the kernel with optimization: shared memory convolution &&
            // loop unrolling && make k a constant memory
            const int H_out = H - K + 1;
            const int W_out = W - K + 1;

            int W_grid = ceil(float(W_out) / TILE_WIDTH);
            int H_grid = ceil(float(H_out) / TILE_WIDTH);
            // int Y = M * C;
            int Wstd = ceil(W_grid / 2.0);
            int Z = Wstd * ceil(H_grid / 2.0);



            // original kernel for ranking
//            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//            int Z = W_grid * ceil(H_grid / 2.0);
//            dim3 gridDim(B, M, Z);
//            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
//            forward_kernel << < gridDim, blockDim >> >
//                                         (y.dptr_, x.dptr_, M, C, H, W, K, W_grid);

            // kernel with shared memory, restricted and const memory
//            dim3 gridDim(B, M, Z);
//            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//
//            int sm_H = TILE_WIDTH + K - 1 + TILE_WIDTH;
//            int sm_W = sm_H;
//
//            size_t shmem_size = sizeof(float) * sm_H * sm_W;
//
//            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
//
//            forward_kernel_optimized_shared_memory_convolution_with_loop_unrolling_with_atomic
//                    << < gridDim, blockDim, shmem_size >> >
//                    (y.dptr_, x.dptr_, M, C, H, W, K, Wstd, sm_H, sm_W);

            // kernel with shared memory only
//            size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
//            forward_kernel_optimized_shared_memory_convolution <<<gridDim, blockDim, shmem_size>>>
//            (k.dptr_, y.dptr_,x.dptr_,M,C,H,W,K,W_grid);


            // the kernel with restricted and const memory and atomicAdd
            dim3 block2(TILE_WIDTH, TILE_WIDTH, 1);
            dim3 grid2(B, M * C, W_grid * H_grid);
            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
            forward_kernel_atomicAdd << < grid2, block2 >> >
            (y.dptr_, x.dptr_, B, M, C, H, W, K, W_grid);

            // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
            // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        }

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
        template<typename gpu, typename DType>
        void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x,
                     const mshadow::Tensor<gpu, 4, DType> &w) {
            CHECK_EQ(0, 1) << "Remove this line and replace it with your implementation.";
        }
    }
}

#endif