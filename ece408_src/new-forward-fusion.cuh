

#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 32 // ???

namespace mxnet {
    namespace op {
        __constant__ float constK[15000];

        __global__ void matrix_multi_unroll_shared_kernal(float *__restrict__ k, float *__restrict__ x,
                float *__restrict__ y, int B, int M, int C, int H, int W, int K,
                int numARows, int numAColumns,
                int numBRows, int numBColumns,
                int numCRows, int numCColumns) {
            //@@ Insert code to implement matrix multiplication here
            //@@ You have to use shared memory for this MP
            __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
            __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;
            int b = blockIdx.z;
            int kSquare = K * K;

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

            int Row = by * TILE_WIDTH + ty;
            int Col = bx * TILE_WIDTH + tx;

            float Pvalue = 0.0;

            int W_out = W - K + 1;
            int H_out = H - K + 1;

#pragma unroll

            for (int i = 0; i < numAColumns / TILE_WIDTH + 1; ++i) {
                if ((Row < numARows) && (i * TILE_WIDTH + tx < numAColumns)) {
                    int k_row = Row;
                    int k_col = i * TILE_WIDTH + tx;
                    int k_m = k_row;
                    int k_c = k_col / kSquare;
                    int k_temp = k_col % kSquare;
                    int k_h = k_temp / K;
                    int k_w = k_temp % K;
                    subTileM[ty][tx] = k4d(k_m, k_c, k_h, k_w);
                } else subTileM[ty][tx] = 0.0;
                __syncthreads();
                if ((i * TILE_WIDTH + ty < numBRows) && (Col < numBColumns)) {
                    int x_row = i * TILE_WIDTH + ty;
                    int x_col = Col;
                    int x_b = b;
                    int x_c = x_row / kSquare;
                    int x_h = x_col / W_out;
                    int x_w = x_col % W_out;
                    int x_temp = x_row % kSquare;
                    int x_p = x_temp / K;
                    int x_q = x_temp % K;
                    subTileN[ty][tx] = x4d(x_b, x_c, x_h + x_p, x_w + x_q);
                } else subTileN[ty][tx] = 0.0;
                __syncthreads();

                if ((Row < numCRows) && (Col < numCColumns)) {
                    for (int k = 0; k < TILE_WIDTH; k++) {
                        Pvalue += subTileM[ty][k] * subTileN[k][tx];
                    }
                }
                __syncthreads();
            }

            if ((Row < numCRows) && (Col < numCColumns)) {
                int y_m = Row;
                int y_h = Col / W_out;
                int y_w = Col % W_out;
                y4d(b, y_m, y_h, y_w) = Pvalue; //
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

            const int B = x.shape_[0];
            const int M = y.shape_[1];
            const int C = x.shape_[1];
            const int H = x.shape_[2];
            const int W = x.shape_[3];
            const int K = k.shape_[3];
            const int H_out = H - K + 1;
            const int W_out = W - K + 1;

            const int numARows = M;    // number of rows in the matrix A
            const int numAColumns = C * K * K; // number of columns in the matrix A
            const int numBRows = C * K * K;    // number of rows in the matrix B
            const int numBColumns = W_out * H_out; // number of columns in the matrix B
            const int numCRows = numARows;    // number of rows in the matrix C (you have to set this)
            const int numCColumns = numBColumns;


            // cudaMemcpyToSymbol(constK, k.dptr_, M * C * K * K * sizeof(float));
            dim3 dimGrid(ceil((1.0 * numCColumns) / TILE_WIDTH), ceil((1.0 * numCRows) / TILE_WIDTH), B);
            dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
            matrix_multi_unroll_shared_kernal << < dimGrid, dimBlock >> >
            (k.dptr_, x.dptr_, y.dptr_, B, M, C, H, W, K,
                    numARows, numAColumns,
                    numBRows, numBColumns,
                    numCRows, numCColumns);

            //MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
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
