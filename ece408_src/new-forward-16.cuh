#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define TILE_WIDTH 8
#define CUDA_MAX_NUM_THREADS 512

namespace mxnet {
    namespace op {

        __constant__ float constK[15000];

        __global__ void forward_kernel(float *__restrict__ y, const float *__restrict__ x,
                                       const int M, const int C, const int H, const int W, const int K,
                                       const int W_grid) {

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            const int h = ((blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y) * 2;
            const int w = ((blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x) * 2;
            const int h2 = h + 1;
            const int w2 = w + 1;

            const bool flag1 = h < H_out && w < W_out;
            const bool flag2 = h < H_out && w2 < W_out;
            const bool flag3 = h2 < H_out && w < W_out;
            const bool flag4 = h2 < H_out && w2 < W_out;

            float acc1 = 0.0;
            float acc2 = 0.0;
            float acc3 = 0.0;
            float acc4 = 0.0;

            const int b = blockIdx.x;
            const int m = blockIdx.y;

            int iter = K / 2;
            if (iter * 2 < K)
                iter += 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

#pragma unroll
            for (int c = 0; c < C; ++c) {
                for (int p = 0; p < iter; ++p) {
                    for (int q = 0; q < iter; ++q) {
                        int p1 = 2 * p;
                        int p2 = 2 * p + 1;

                        int q1 = 2 * q;
                        int q2 = 2 * q + 1;

                        if (flag1) acc1 += x4d(b, c, h + p1, w + q1) * k4d(m, c, p1, q1);
                        if (flag2) acc2 += x4d(b, c, h + p1, w2 + q1) * k4d(m, c, p1, q1);
                        if (flag3) acc3 += x4d(b, c, h2 + p1, w + q1) * k4d(m, c, p1, q1);
                        if (flag4) acc4 += x4d(b, c, h2 + p1, w2 + q1) * k4d(m, c, p1, q1);

                        if (q2 < K && p2 < K) {
                            if (flag1)
                                acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2) +
                                        x4d(b, c, h + p2, w + q2) * k4d(m, c, p2, q2) +
                                        x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                            if (flag2)
                                acc2 += x4d(b, c, h + p1, w2 + q2) * k4d(m, c, p1, q2) +
                                        x4d(b, c, h + p2, w2 + q2) * k4d(m, c, p2, q2) +
                                        x4d(b, c, h + p2, w2 + q1) * k4d(m, c, p2, q1);
                            if (flag3)
                                acc3 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2) +
                                        x4d(b, c, h2 + p2, w + q2) * k4d(m, c, p2, q2) +
                                        x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
                            if (flag4)
                                acc4 += x4d(b, c, h2 + p1, w2 + q2) * k4d(m, c, p1, q2) +
                                        x4d(b, c, h2 + p2, w2 + q2) * k4d(m, c, p2, q2) +
                                        x4d(b, c, h2 + p2, w2 + q1) * k4d(m, c, p2, q1);
                        } else if (q2 < K) {
                            if (flag1) acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2);
                            if (flag2) acc2 += x4d(b, c, h + p1, w2 + q2) * k4d(m, c, p1, q2);
                            if (flag3) acc3 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2);
                            if (flag4) acc4 += x4d(b, c, h2 + p1, w2 + q2) * k4d(m, c, p1, q2);
                        } else if (p2 < K) {
                            if (flag1) acc1 += x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                            if (flag2) acc2 += x4d(b, c, h + p2, w2 + q1) * k4d(m, c, p2, q1);
                            if (flag3) acc3 += x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
                            if (flag4) acc4 += x4d(b, c, h2 + p2, w2 + q1) * k4d(m, c, p2, q1);
                        }
                    }
                }
            }

            if (flag1) y4d(b, m, h, w) = acc1;
            if (flag2) y4d(b, m, h, w2) = acc2;
            if (flag3) y4d(b, m, h2, w) = acc3;
            if (flag4) y4d(b, m, h2, w2) = acc4;

#undef y4d
#undef x4d
#undef k4d
        }

        __global__ void forward_kernel_atomicAdd(float *__restrict__ y, const float *__restrict__ x,
                                                 const int B, const int M, const int C, const int H,
                                                 const int W, const int K, const int W_grid) {

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            const int h = ((blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y) * 4;
            const int w = ((blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x) * 2;

            int iter = K / 2;
            if (iter * 2 < K)
                iter += 1;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            int b = blockIdx.x;
            int m = blockIdx.y % M;
            int c = blockIdx.y / M;
            int h2 = h + 1;
            int h3 = h + 2;
            int h4 = h + 3;
            int w2 = w + 1;
            const bool flag1 = h < H_out && w < W_out;
            const bool flag2 = h < H_out && w2 < W_out;
            const bool flag3 = h2 < H_out && w < W_out;
            const bool flag4 = h2 < H_out && w2 < W_out;
            const bool flag5 = h3 < H_out && w < W_out;
            const bool flag6 = h3 < H_out && w2 < W_out;
            const bool flag7 = h4 < H_out && w < W_out;
            const bool flag8 = h4 < H_out && w2 < W_out;

            float acc1 = 0.0;
            float acc2 = 0.0;
            float acc3 = 0.0;
            float acc4 = 0.0;
            float acc5 = 0.0;
            float acc6 = 0.0;
            float acc7 = 0.0;
            float acc8 = 0.0;


#pragma unroll

            for (int p = 0; p < iter; ++p) {
                for (int q = 0; q < iter; ++q) {
                    int p1 = 2 * p;
                    int p2 = 2 * p + 1;

                    int q1 = 2 * q;
                    int q2 = 2 * q + 1;

                    if (flag1) acc1 += x4d(b, c, h + p1, w + q1) * k4d(m, c, p1, q1);
                    if (flag2) acc2 += x4d(b, c, h + p1, w2 + q1) * k4d(m, c, p1, q1);
                    if (flag3) acc3 += x4d(b, c, h2 + p1, w + q1) * k4d(m, c, p1, q1);
                    if (flag4) acc4 += x4d(b, c, h2 + p1, w2 + q1) * k4d(m, c, p1, q1);
                    if (flag5) acc5 += x4d(b, c, h3 + p1, w + q1) * k4d(m, c, p1, q1);
                    if (flag6) acc6 += x4d(b, c, h3 + p1, w2 + q1) * k4d(m, c, p1, q1);
                    if (flag7) acc7 += x4d(b, c, h4 + p1, w + q1) * k4d(m, c, p1, q1);
                    if (flag8) acc8 += x4d(b, c, h4 + p1, w2 + q1) * k4d(m, c, p1, q1);

                    if (q2 < K && p2 < K) {
                        if (flag1)
                            acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h + p2, w + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                        if (flag2)
                            acc2 += x4d(b, c, h + p1, w2 + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h + p2, w2 + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h + p2, w2 + q1) * k4d(m, c, p2, q1);
                        if (flag3)
                            acc3 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h2 + p2, w + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
                        if (flag4)
                            acc4 += x4d(b, c, h2 + p1, w2 + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h2 + p2, w2 + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h2 + p2, w2 + q1) * k4d(m, c, p2, q1);
                        if (flag5)
                            acc5 += x4d(b, c, h3 + p1, w + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h3 + p2, w + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h3 + p2, w + q1) * k4d(m, c, p2, q1);
                        if (flag6)
                            acc6 += x4d(b, c, h3 + p1, w2 + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h3 + p2, w2 + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h3 + p2, w2 + q1) * k4d(m, c, p2, q1);
                        if (flag7)
                            acc7 += x4d(b, c, h4 + p1, w + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h4 + p2, w + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h4 + p2, w + q1) * k4d(m, c, p2, q1);
                        if (flag8)
                            acc8 += x4d(b, c, h4 + p1, w2 + q2) * k4d(m, c, p1, q2) +
                                    x4d(b, c, h4 + p2, w2 + q2) * k4d(m, c, p2, q2) +
                                    x4d(b, c, h4 + p2, w2 + q1) * k4d(m, c, p2, q1);
                    } else if (q2 < K) {
                        if (flag1) acc1 += x4d(b, c, h + p1, w + q2) * k4d(m, c, p1, q2);
                        if (flag2) acc2 += x4d(b, c, h + p1, w2 + q2) * k4d(m, c, p1, q2);
                        if (flag3) acc3 += x4d(b, c, h2 + p1, w + q2) * k4d(m, c, p1, q2);
                        if (flag4) acc4 += x4d(b, c, h2 + p1, w2 + q2) * k4d(m, c, p1, q2);
                        if (flag5) acc5 += x4d(b, c, h3 + p1, w + q2) * k4d(m, c, p1, q2);
                        if (flag6) acc6 += x4d(b, c, h3 + p1, w2 + q2) * k4d(m, c, p1, q2);
                        if (flag7) acc7 += x4d(b, c, h4 + p1, w + q2) * k4d(m, c, p1, q2);
                        if (flag8) acc8 += x4d(b, c, h4 + p1, w2 + q2) * k4d(m, c, p1, q2);
                    } else if (p2 < K) {
                        if (flag1) acc1 += x4d(b, c, h + p2, w + q1) * k4d(m, c, p2, q1);
                        if (flag2) acc2 += x4d(b, c, h + p2, w2 + q1) * k4d(m, c, p2, q1);
                        if (flag3) acc3 += x4d(b, c, h2 + p2, w + q1) * k4d(m, c, p2, q1);
                        if (flag4) acc4 += x4d(b, c, h2 + p2, w2 + q1) * k4d(m, c, p2, q1);
                        if (flag5) acc5 += x4d(b, c, h3 + p2, w + q1) * k4d(m, c, p2, q1);
                        if (flag6) acc6 += x4d(b, c, h3 + p2, w2 + q1) * k4d(m, c, p2, q1);
                        if (flag7) acc7 += x4d(b, c, h4 + p2, w + q1) * k4d(m, c, p2, q1);
                        if (flag8) acc8 += x4d(b, c, h4 + p2, w2 + q1) * k4d(m, c, p2, q1);
                    }
                }
            }

            if (flag1) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h * W_out + w], acc1);
            if (flag2) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h * W_out + w2], acc2);
            if (flag3) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h2 * W_out + w], acc3);
            if (flag4) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h2 * W_out + w2], acc4);
            if (flag5) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h3 * W_out + w], acc5);
            if (flag6) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h3 * W_out + w2], acc6);
            if (flag7) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h4 * W_out + w], acc7);
            if (flag8) atomicAdd(&y[b * M * H_out * W_out + m * H_out * W_out + h4 * W_out + w2], acc8);

#undef y4d
#undef x4d
#undef k4d
        }

        __global__ void forward_kernel_optimized_shared_memory_convolution_with_loop_unrolling
                (float *__restrict__ y, const float *__restrict__ x, const int M,
                 const int C, const int H, const int W, const int K, const int W_grid,
                 const int sm_H, const int sm_W) {

            const int H_out = H - K + 1;
            const int W_out = W - K + 1;
            extern __shared__ float shmem[];

            float *X_shared  = &shmem[0];

            const int b = blockIdx.x;
            const int m = blockIdx.y;
            const int h = (blockIdx.z / W_grid) * TILE_WIDTH * 4 + threadIdx.y;
            const int h2 = h  + TILE_WIDTH;
            const int h3 = h2 + TILE_WIDTH;
            const int h4 = h3 + TILE_WIDTH;
            const int w = (blockIdx.z % W_grid) * TILE_WIDTH * 4 + threadIdx.x;
            const int w2 = w  + TILE_WIDTH;
            const int w3 = w2 + TILE_WIDTH;
            const int w4 = w3 + TILE_WIDTH;

            float acc  = 0.0;
            float acc2 = 0.0;
            float acc3 = 0.0;
            float acc4 = 0.0;
            float acc5 = 0.0;
            float acc6 = 0.0;
            float acc7 = 0.0;
            float acc8 = 0.0;
            float acc9 = 0.0;
            float acc10 = 0.0;
            float acc11 = 0.0;
            float acc12 = 0.0;
            float acc13 = 0.0;
            float acc14 = 0.0;
            float acc15 = 0.0;
            float acc16 = 0.0;
//
//            int iter = K / 2;
//            if (iter * 2 < K) {
//                iter += 1;
//            }

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x2d(i1, i0)  X_shared[i1 * sm_W + i0]
#define k4d(i3, i2, i1, i0) constK[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

#pragma unroll

            for (int c = 0; c < C; c++) {
                for (int i = 0; i < sm_H-threadIdx.y; i += TILE_WIDTH) {
                    for (int j = 0; j < sm_W-threadIdx.x; j += TILE_WIDTH) {
                        x2d ((threadIdx.y+i), (threadIdx.x+j)) = x4d(b,c,(h+i), (w+j));
                    }
                }
                __syncthreads();

                for (int p = 0; p < K; ++p) {
                    for (int q = 0; q < K; ++q) {
                        float mask = k4d(m,c,p,q);
                        int xPos1 = threadIdx.x + q;
                        int xPos2 = threadIdx.x + 1 * TILE_WIDTH + q;
                        int xPos3 = threadIdx.x + 2 * TILE_WIDTH + q;
                        int xPos4 = threadIdx.x + 3 * TILE_WIDTH + q;
                        int yPos1 = threadIdx.y + p;
                        int yPos2 = threadIdx.y + 1 * TILE_WIDTH + p;
                        int yPos3 = threadIdx.y + 2 * TILE_WIDTH + p;
                        int yPos4 = threadIdx.y + 3 * TILE_WIDTH + p;
                        acc  += x2d (yPos1, xPos1) * mask;
                        acc2 += x2d (yPos1, xPos2) * mask;
                        acc3 += x2d (yPos1, xPos3) * mask;
                        acc4 += x2d (yPos1, xPos4) * mask;

                        acc5 += x2d (yPos2, xPos1) * mask;
                        acc6 += x2d (yPos2, xPos2) * mask;
                        acc7 += x2d (yPos2, xPos3) * mask;
                        acc8 += x2d (yPos2, xPos4) * mask;

                        acc9  += x2d (yPos3, xPos1) * mask;
                        acc10 += x2d (yPos3, xPos2) * mask;
                        acc11 += x2d (yPos3, xPos3) * mask;
                        acc12 += x2d (yPos3, xPos4) * mask;

                        acc13 += x2d (yPos4, xPos1) * mask;
                        acc14 += x2d (yPos4, xPos2) * mask;
                        acc15 += x2d (yPos4, xPos3) * mask;
                        acc16 += x2d (yPos4, xPos4) * mask;
                    }
                }
                __syncthreads();
            }

            if (h < H_out && w < W_out) y4d(b, m, h, w) = acc;

            if (h < H_out && w2 < W_out) y4d(b, m, h, w2) = acc2;

            if (h < H_out && w3 < W_out) y4d(b, m, h, w3) = acc3;

            if (h < H_out && w4 < W_out) y4d(b, m, h, w4) = acc4;

            if (h2 < H_out && w < W_out) y4d(b, m, h2, w) = acc5;

            if (h2 < H_out && w2 < W_out) y4d(b, m, h2, w2) = acc6;

            if (h2 < H_out && w3 < W_out) y4d(b, m, h2, w3) = acc7;

            if (h2 < H_out && w4 < W_out) y4d(b, m, h2, w4) = acc8;

            if (h3 < H_out && w < W_out) y4d(b, m, h3, w) = acc9;

            if (h3 < H_out && w2 < W_out) y4d(b, m, h3, w2) = acc10;

            if (h3 < H_out && w3 < W_out) y4d(b, m, h3, w3) = acc11;

            if (h3 < H_out && w4 < W_out) y4d(b, m, h3, w4) = acc12;

            if (h4 < H_out && w < W_out) y4d(b, m, h4, w) = acc13;

            if (h4 < H_out && w2 < W_out) y4d(b, m, h4, w2) = acc14;

            if (h4 < H_out && w3 < W_out) y4d(b, m, h4, w3) = acc15;

            if (h4 < H_out && w4 < W_out) y4d(b, m, h4, w4) = acc16;
#undef y4d
#undef x4d
#undef x2d
#undef k4d
        }

        __global__ void forward_kernel_optimized_shared_memory_convolution
                (float *k, float *y, const float *x, const int M,
                 const int C, const int H, const int W, const int K, const int W_grid) {
            const int H_out = H - K + 1;
            const int W_out = W - K + 1;

            int X_tile_width = TILE_WIDTH + K - 1;
            extern __shared__ float shmem[];

            float *X_shared = &shmem[0];
            float *K_shared = X_shared + X_tile_width * X_tile_width;

            int b = blockIdx.x;
            int m = blockIdx.y;

            int h0 = threadIdx.y;
            int w0 = threadIdx.x;

            int h_base = (blockIdx.z / W_grid) * TILE_WIDTH;
            int w_base = (blockIdx.z % W_grid) * TILE_WIDTH;

            int h = h_base + h0;
            int w = w_base + w0;

            float acc = 0.0;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            for (int c = 0; c < C; c++) {

                if (h0 < K && w0 < K) {
                    K_shared[h0 * K + w0] = k4d(m, c, h0, w0);
                }
                __syncthreads();

                for (int i = h0; i < X_tile_width; i += TILE_WIDTH) {
                    for (int j = w0; j < X_tile_width; j += TILE_WIDTH) {
                        X_shared[i * X_tile_width + j] = x4d(b, c, i + h_base, j + w_base);
                    }
                }
                __syncthreads();

                for (int p = 0; p < K; ++p) {
                    for (int q = 0; q < K; ++q) {
                        int pos_x = (h0 + p) * X_tile_width + (w0 + q);
                        int pos_k = p * K + q;
                        acc += X_shared[pos_x] * K_shared[pos_k];
                    }
                }
                __syncthreads();
            }

            if (h < H_out && w < W_out) {
                y4d(b, m, h, w) = acc;
            }

#undef y4d
#undef x4d
#undef k4d
        }


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



            // original kernel for ranking
//            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
//            int Z = ceil(W_grid/2.0) * ceil(H_grid / 2.0);
//            dim3 gridDim(B, M, Z);
//            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
//            forward_kernel << < gridDim, blockDim >> >
//                                         (y.dptr_, x.dptr_, M, C, H, W, K, ceil(W_grid/2.0));

            // kernel with shared memory, restricted and const memory
            dim3 gridDim(B, M, ceil(W_grid/4.0) * ceil(H_grid/4.0));
            dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

            int sm_H = TILE_WIDTH + K - 1 + 3 * TILE_WIDTH;
            int sm_W = TILE_WIDTH + K - 1 + 3 * TILE_WIDTH;

            size_t shmem_size = sizeof(float) * sm_H * sm_W;

            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));

            forward_kernel_optimized_shared_memory_convolution_with_loop_unrolling
                    << < gridDim, blockDim, shmem_size >> >
                    (y.dptr_, x.dptr_, M, C, H, W, K, ceil(W_grid/4.0), sm_H, sm_W);

            // kernel with shared memory only
//            size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
//            forward_kernel_optimized_shared_memory_convolution <<<gridDim, blockDim, shmem_size>>>
//            (k.dptr_, y.dptr_,x.dptr_,M,C,H,W,K,W_grid);


            // the kernel with restricted and const memory and atomicAdd
//            dim3 block2(TILE_WIDTH, TILE_WIDTH, 1);
//            dim3 grid2(B, M * C, ceil(W_grid / 2.0) * ceil(H_grid / 4.0));
//            cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
//            forward_kernel_atomicAdd << < grid2, block2 >> >
//            (y.dptr_, x.dptr_, B, M, C, H, W, K, ceil(W_grid / 2.0));

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