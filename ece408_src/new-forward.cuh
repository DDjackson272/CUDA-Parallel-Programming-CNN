#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16

namespace mxnet {
    namespace op {
        __constant__ float constK[15000];

        // code for fusion
//        __global__ void matrix_multi_unroll_shared_kernal(float *__restrict__ k, float *__restrict__ x,
//                float *__restrict__ y, int B, int M, int C, int H, int W, int K,
//                int numARows, int numAColumns,
//                int numBRows, int numBColumns,
//                int numCRows, int numCColumns) {
//            //@@ Insert code to implement matrix multiplication here
//            //@@ You have to use shared memory for this MP
//            __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
//            __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
//
//            int bx = blockIdx.x;
//            int by = blockIdx.y;
//            int tx = threadIdx.x;
//            int ty = threadIdx.y;
//            int b = blockIdx.z;
//            int kSquare = K * K;
//
//#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//            int Row = by * TILE_WIDTH + ty;
//            int Col = bx * TILE_WIDTH + tx;
//
//            float Pvalue = 0.0;
//
//            int W_out = W - K + 1;
//            int H_out = H - K + 1;
//
//#pragma unroll
//
//            for (int i = 0; i < numAColumns / TILE_WIDTH + 1; ++i) {
//                if ((Row < numARows) && (i * TILE_WIDTH + tx < numAColumns)) {
//                    int k_row = Row;
//                    int k_col = i * TILE_WIDTH + tx;
//                    int k_m = k_row;
//                    int k_c = k_col / kSquare;
//                    int k_temp = k_col % kSquare;
//                    int k_h = k_temp / K;
//                    int k_w = k_temp % K;
//                    subTileM[ty][tx] = k4d(k_m, k_c, k_h, k_w);
//                } else subTileM[ty][tx] = 0.0;
//                __syncthreads();
//                if ((i * TILE_WIDTH + ty < numBRows) && (Col < numBColumns)) {
//                    int x_row = i * TILE_WIDTH + ty;
//                    int x_col = Col;
//                    int x_b = b;
//                    int x_c = x_row / kSquare;
//                    int x_h = x_col / W_out;
//                    int x_w = x_col % W_out;
//                    int x_temp = x_row % kSquare;
//                    int x_p = x_temp / K;
//                    int x_q = x_temp % K;
//                    subTileN[ty][tx] = x4d(x_b, x_c, x_h + x_p, x_w + x_q);
//                } else subTileN[ty][tx] = 0.0;
//                __syncthreads();
//
//                if ((Row < numCRows) && (Col < numCColumns)) {
//                    for (int k = 0; k < TILE_WIDTH; k++) {
//                        Pvalue += subTileM[ty][k] * subTileN[k][tx];
//                    }
//                }
//                __syncthreads();
//            }
//
//            if ((Row < numCRows) && (Col < numCColumns)) {
//                int y_m = Row;
//                int y_h = Col / W_out;
//                int y_w = Col % W_out;
//                y4d(b, y_m, y_h, y_w) = Pvalue; //
//            }
//
//#undef y4d
//#undef x4d
//#undef k4d
//        }

// code for ranking
        __global__ void forward_kernel(float *__restrict__ y, const float *__restrict__ x,
                                       const int M, const int C, const int H, const int W, const int K,
                                       const int W_grid) {

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            const int h = ((blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y) * 2;
            const int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
            const int h2 = h + 1;

            int iter = K / 2;
            if (iter * 2 < K)
                iter += 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            int b = blockIdx.x;
            int m = blockIdx.y;

            if (w < W_out && h2 < H_out) {
                float acc = 0.0;
                float acc2 = 0.0;
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    for (int p = 0; p < iter; ++p) {
                        for (int q = 0; q < iter; ++q) {
                            int p1 = 2 * p;
                            int p2 = 2 * p + 1;

                            int q1 = 2 * q;
                            int q2 = 2 * q + 1;

                            acc += x4d(b, c, h + p1, w + q1) * k4d(m, c, p1, q1);
                            acc2 += x4d(b, c, h2 + p1, w + q1) * k4d(m, c, p1, q1);

                            if (q2 < K && p2 < K) {
                                acc += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2) +
                                       x4d(b, c, h + p2, w + q2) * k4d(m, c, p2, q2) +
                                       x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                                acc2 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2) +
                                        x4d(b, c, h2 + p2, w + q2) * k4d(m, c, p2, q2) +
                                        x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
                            } else if (q2 < K) {
                                acc += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2);
                                acc2 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2);
                            } else if (p2 < K) {
                                acc += x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                                acc2 += x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
                            }
                        }
                    }
                }
                y4d(b, m, h, w) = acc;
                y4d(b, m, h2, w) = acc2;
            } else if (h < H_out && w < W_out) {
                float acc = 0.0;
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    for (int p = 0; p < iter; ++p) {
                        for (int q = 0; q < iter; ++q) {
                            int p1 = 2 * p;
                            int p2 = 2 * p + 1;

                            int q1 = 2 * q;
                            int q2 = 2 * q + 1;

                            acc += x4d(b, c, h + p1, w + q1) * k4d(m, c, p1, q1);

                            if (q2 < K && p2 < K) {
                                acc += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2) +
                                       x4d(b, c, h + p2, w + q2) * k4d(m, c, p2, q2) +
                                       x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                            } else if (q2 < K) {
                                acc += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2);
                            } else if (p2 < K) {
                                acc += x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                            }
                        }
                    }
                }
                y4d(b, m, h, w) = acc;
            } else if (h2 < H_out && w < W_out) {
                float acc2 = 0.0;
#pragma unroll
                for (int c = 0; c < C; ++c) {
                    for (int p = 0; p < iter; ++p) {
                        for (int q = 0; q < iter; ++q) {
                            int p1 = 2 * p;
                            int p2 = 2 * p + 1;

                            int q1 = 2 * q;
                            int q2 = 2 * q + 1;

                            acc2 += x4d(b, c, h2 + p1, w + q1) * k4d(m, c, p1, q1);

                            if (q2 < K && p2 < K) {
                                acc2 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2) +
                                        x4d(b, c, h2 + p2, w + q2) * k4d(m, c, p2, q2) +
                                        x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
                            } else if (q2 < K) {
                                acc2 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2);
                            } else if (p2 < K) {
                                acc2 += x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
                            }
                        }
                    }
                }
                y4d(b, m, h2, w) = acc2;
            }
#undef y4d
#undef x4d
#undef k4d
        }

        // code for atomic operation

//        __global__ void forward_kernel_atomicAdd(float *__restrict__ y, const float *__restrict__ x,
//                const int B, const int M, const int C, const int H, const int W, const int K,
//                const int W_grid) {
//
//            const int H_out = H - K + 1;
//            const int W_out = W - K + 1;
//            const int h_base = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
//            const int w_base = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
//
//            int iter = K / 2;
//            if (iter * 2 < K)
//                iter += 1;
//
//#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//
//            int b = blockIdx.x;
//            int m = blockIdx.y % M;
//            int c = blockIdx.y / M;
//            int h = h_base;
//            int w = w_base;
//            if (h < H_out && w < W_out) {
//                float acc = 0.0;
//#pragma unroll
//                for (int p = 0; p < iter; ++p) {
//                    for (int q = 0; q < iter; ++q) {
//                        int p1 = 2 * p;
//                        int p2 = 2 * p + 1;
//
//                        int q1 = 2 * q;
//                        int q2 = 2 * q + 1;
//
//                        acc += x4d(b, c, h + p1, w + q1) * k4d(m, c, p1, q1);
//
//                        if (q2 < K && p2 < K) {
//                            acc += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2) +
//                                   x4d(b, c, h + p2, w + q2) * k4d(m, c, p2, q2) +
//                                   x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
//                        } else if (q2 < K) {
//                            acc += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2);
//                        } else if (p2 < K) {
//                            acc += x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
//                        }
//                    }
//                }
//                atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h * W_out + w], acc);
//            }
//
//#undef y4d
//#undef x4d
//#undef k4d
//        }

//        __global__ void forward_kernel_optimized_shared_memory_convolution_with_loop_unrolling
//                (float *__restrict__ y, const float *__restrict__ x, const int M,
//                 const int C, const int H, const int W, const int K, const int W_grid) {
//            const int H_out = H - K + 1;
//            const int W_out = W - K + 1;
//
//            int X_tile_width = TILE_WIDTH + K - 1;
//            extern __shared__ float shmem[];
//
//            float *X_shared = &shmem[0];
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
//            int iter = K / 2;
//            if (iter * 2 < K) {
//                iter += 1;
//            }
//
//#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
//#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
//
//#pragma unroll
//
//            float mask11 = constK[m * (C * K * K) + c * (K * K) + p1 * (K) + q1];
//            float mask12 = constK[m * (C * K * K) + c * (K * K) + p1 * (K) + q2];
//            float mask21 = constK[m * (C * K * K) + c * (K * K) + p2 * (K) + q1];
//            float mask22 = constK[m * (C * K * K) + c * (K * K) + p2 * (K) + q2];
//
//            for (int c = 0; c < C; c++) {
//                for (int i = h0; i < X_tile_width; i += TILE_WIDTH) {
//                    for (int j = w0; j < X_tile_width; j += TILE_WIDTH) {
//                        X_shared[i * X_tile_width + j] = x4d(b, c, i + h_base, j + w_base);
//                    }
//                }
//                __syncthreads();
//
//                for (int p = 0; p < iter; ++p) {
//                    for (int q = 0; q < iter; ++q) {
//
//                        int p1 = 2 * p;
//                        int p2 = 2 * p + 1;
//
//                        int q1 = 2 * q;
//                        int q2 = 2 * q + 1;
//
//                        acc += X_shared[(h0 + p1) * X_tile_width + (w0 + q1)] * mask11;
//
//                        if (p2 < K && q2 < K) {
//                            acc += X_shared[(h0 + p1) * X_tile_width + (w0 + q2)] * mask12;
//
//                            acc += X_shared[(h0 + p2) * X_tile_width + (w0 + q2)] * mask22;
//
//                            acc += X_shared[(h0 + p2) * X_tile_width + (w0 + q1)] * mask21;
//                        } else if (p2 < K) {
//                            acc += X_shared[(h0 + p2) * X_tile_width + (w0 + q1)] * mask21;
//                        } else if (q2 < K) {
//                            acc += X_shared[(h0 + p1) * X_tile_width + (w0 + q2)] * mask12;
//                        }
//                    }
//                }
//                __syncthreads();
//            }
//
//            if (h < H_out && w < W_out) {
//                y4d(b, m, h, w) = acc;
//            }
//#undef y4d
//#undef x4d
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

            const int numARows = M;    // number of rows in the matrix A
            const int numAColumns = C * K * K; // number of columns in the matrix A
            const int numBRows = C * K * K;    // number of rows in the matrix B
            const int numBColumns = W_out * H_out; // number of columns in the matrix B
            const int numCRows = numARows;    // number of rows in the matrix C (you have to set this)
            const int numCColumns = numBColumns;

            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
            int W_grid = ceil(float(W_out) / TILE_WIDTH);
            int H_grid = ceil(float(H_out) / TILE_WIDTH);
            int Z = W_grid * ceil(H_grid / 2.0);
            dim3 gridDim(B, M, Z);

            // original kernel for ranking
            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
            forward_kernel << < gridDim, blockDim >> > (y.dptr_, x.dptr_, M, C, H, W, K, W_grid);

            // kernel with shared memory, restricted and const memory
//            size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1));
//            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
//            forward_kernel_optimized_shared_memory_convolution_with_loop_unrolling
//            <<<gridDim, blockDim, shmem_size>>>
//            (y.dptr_,x.dptr_,M,C,H,W,K,W_grid);

            // kernel with shared memory only
//            size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
//            forward_kernel_optimized_shared_memory_convolution <<<gridDim, blockDim, shmem_size>>>
//            (k.dptr_, y.dptr_,x.dptr_,M,C,H,W,K,W_grid);


            // the kernel with restricted and const memory and atomicAdd
//            dim3 block2(TILE_WIDTH, TILE_WIDTH,1);
//            dim3 grid2(B, M*C, W_grid * ceil(H_grid/2.0));
//            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
//            forward_kernel_atomicAdd<<<grid2, block2>>>(y.dptr_,x.dptr_,B,M,C,H,W,K,W_grid);

//            dim3 dimGrid(ceil((1.0 * numCColumns) / TILE_WIDTH), ceil((1.0 * numCRows) / TILE_WIDTH), B);
//            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
//            matrix_multi_unroll_shared_kernal << < dimGrid, dimBlock >> >
//                                                            (k.dptr_, x.dptr_, y.dptr_, B, M, C, H, W, K,
//                                                                    numARows, numAColumns,
//                                                                    numBRows, numBColumns,
//                                                                    numCRows, numCColumns);

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