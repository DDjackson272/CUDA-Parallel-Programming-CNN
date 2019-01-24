

#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 16 // ???
#define MAX_NUM_THREADS 512 // ???
namespace mxnet
{
    namespace op
    {


        __global__ void unroll_kernel(int C, int H, int W, int K, int b, float* x, float* x_unroll){
            int c, s, h_out_index, w_out_index, h_unroll, w_unroll, h_base, p, q;
            int t = blockIdx.x * blockDim.x + threadIdx.x;
            int W_x_unroll = (H - K + 1) * (W - K + 1);

#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define x_unroll2d(i1, i0) x_unroll[(i1) * W_x_unroll + i0]

            if (t < C * W_x_unroll) {
                c = t / W_x_unroll;
                s = t % W_x_unroll;
                h_out_index = s / (W - K + 1);
                w_out_index = s % (W - K + 1);
                w_unroll = s;
                h_base = c*K*K;
                for (p = 0; p < K; p++) {
                    for (q = 0; q < K; q++) {
                        h_unroll = h_base + p*K + q; // ???
                        x_unroll2d(h_unroll, w_unroll) = x4d(b, c, h_out_index+p, w_out_index+q);
                    }
                }
            }

#undef x4d
#undef x_unroll2d
        }


        __global__ void matrixMultiply(float *A, float *B, float *C, int b,
                int numARows, int numAColumns, int numBRows, int numBColumns,
                int numCRows, int numCColumns) {

            int Row = blockIdx.y*blockDim.y+threadIdx.y;
            int Col = blockIdx.x*blockDim.x+threadIdx.x;

            if ((Row < numCRows) && (Col < numCColumns)) {
                float Pvalue = 0;
                for (int k = 0; k < numAColumns; ++k) Pvalue += A[Row*numAColumns+k] * B[k*numBColumns+Col];
                C[b*numCRows*numCColumns + Row*numCColumns + Col] = Pvalue;
            }
        }

        __global__ void matrix_multi_shared_kernel(float *A, float *B, float *C, int b,
                int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {

            __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
            __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];

            int bx = blockIdx.x; int by = blockIdx.y;
            int tx = threadIdx.x; int ty = threadIdx.y;

            int Row = by * TILE_WIDTH + ty;
            int Col = bx * TILE_WIDTH + tx;
            float Pvalue = 0.0;

            for (int m = 0; m < numAColumns/TILE_WIDTH + 1; ++m) {
                // Collaborative loading of M and N tiles into shared memory
                if ((Row < numARows) && (m*TILE_WIDTH+tx < numAColumns))  subTileM[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH + tx];
                else subTileM[ty][tx] = 0.0;
                if ((m*TILE_WIDTH+ty < numBRows) && (Col < numBColumns))  subTileN[ty][tx] = B[(m*TILE_WIDTH+ty)* numBColumns + Col]; // ???
                else subTileN[ty][tx] = 0.0;
                __syncthreads();
                for (int k=0; k<TILE_WIDTH; k++) {
                    Pvalue += subTileM[ty][k] * subTileN[k][tx];
                }
                __syncthreads();
            }
            if ((Row < numCRows) && (Col < numCColumns))  C[b*numCRows*numCColumns + Row*numCColumns + Col] = Pvalue; // ???
        }


/*
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
        template <>
        void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y,
                                 const mshadow::Tensor<gpu, 4, float> &x,
                                 const mshadow::Tensor<gpu, 4, float> &k)
        {

            const int B = x.shape_[0];
            const int M = y.shape_[1];
            const int C = x.shape_[1];
            const int H = x.shape_[2];
            const int W = x.shape_[3];
            const int K = k.shape_[3];
            const int H_out = H-K+1;
            const int W_out = W-K+1;
            const int H_unroll = H_out*W_out;
            const int W_unroll = C*K*K;
            const int numARows = M;    // number of rows in the matrix A
            const int numAColumns = C*K*K; // number of columns in the matrix A
            const int numBRows = C*K*K;    // number of rows in the matrix B
            const int numBColumns = W_out*H_out; // number of columns in the matrix B
            const int numCRows = numARows;    // number of rows in the matrix C (you have to set this)
            const int numCColumns = numBColumns;
            float* device_x_unroll;
            cudaMalloc((void**)&device_x_unroll, W_unroll*H_unroll*sizeof(float));

            for (int b = 0; b < B; b++) {
                int H_out = H - K + 1;
                int W_out = H - K + 1;
                int num_threads = C * H_out * W_out;
                dim3 blockDim1(MAX_NUM_THREADS, 1, 1);
                dim3 gridDim1(ceil((1.0*num_threads)/MAX_NUM_THREADS), 1, 1);
                unroll_kernel<<<gridDim1, blockDim1>>>(C, H, W, K, b, x.dptr_, device_x_unroll);
                dim3 dimGrid2(ceil((1.0*numCColumns)/TILE_WIDTH), ceil((1.0*numCRows)/TILE_WIDTH), 1);
                dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
                matrix_multi_shared_kernel<<<dimGrid2, dimBlock2>>>
                (k.dptr_, device_x_unroll, y.dptr_, b, numARows, numAColumns,
                        numBRows, numBColumns,
                        numCRows, numCColumns);
            }

            // MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
        }

/*
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
        template <typename gpu, typename DType>
        void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
        {
            CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
        }
    }
}

#endif
