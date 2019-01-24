

#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define TILE_WIDTH 8 // ???
#define MAX_NUM_THREADS 512 // ???
namespace mxnet
{
namespace op
{

    // __global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M,
    //     const int C, const int H, const int W, const int K, const int W_grid)
    // {
    //
    //
    //     const int H_out = H - K + 1;
    //     const int W_out = W - K + 1;
    //
    // // An example use of these macros:
    // // float a = y4d(0,0,0,0)
    // // y4d(0,0,0,0) = a
    // #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    // #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    //
    //     int b = blockIdx.x;
    //     int m = blockIdx.y;
    //     int h = (blockIdx.z / W_grid) * TILE_WIDTH + threadIdx.y;
    //     int w = (blockIdx.z % W_grid) * TILE_WIDTH + threadIdx.x;
    //     if (0 <= h && h < H_out &&
    //         0 <= w && w < W_out){
    //         float acc = 0.0;
    //         for (int c = 0; c < C; ++c){
    //             for (int p = 0; p < K; ++p){
    //                 for (int q = 0; q < K; ++q){
    //                     float x_element = x4d(b,c,h+p,w+q);
    //                     float k_element = k4d(m,c,p,q);
    //                     acc += x_element*k_element;
    //                 }
    //             }
    //         }
    //         y4d(b,m,h,w) = acc;
    //     }
    // #undef y4d
    // #undef x4d
    // #undef k4d
    // }
    //
    // __global__ void forward_kernel_optimized(float *y, const float *x, const float *k, const int B, const int M,
    //                                    const int C, const int H, const int W, const int K, const int W_grid)
    // {
    //     const int H_out = H - K + 1;
    //     const int W_out = W - K + 1;
    //
    //     int X_tile_width = TILE_WIDTH + K - 1;
    //     extern __shared__ float shmem[];
    //
    //     float *X_shared = &shmem[0]; // size of input data is X_tile_width * X_tile_width;
    //     float *K_shared = X_shared + X_tile_width * X_tile_width; // size of mask is K * K;
    //
    //     int b = blockIdx.x;
    //     int m = blockIdx.y;
    //
    //     int h0 = threadIdx.y;
    //     int w0 = threadIdx.x;
    //
    //     int h_base = (blockIdx.z/W_grid) * TILE_WIDTH;
    //     int w_base = (blockIdx.z%W_grid) * TILE_WIDTH;
    //
    //     int h = h_base + h0;
    //     int w = w_base + w0;
    //
    //     float acc = 0.0;
    //
    // // An example use of these macros:
    // // float a = y4d(0,0,0,0)
    // // y4d(0,0,0,0) = a
    // #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    // #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    // #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define k_shared2d(i1, i0)  K_shared[i1 * K + i0] // size is K * K
    // #define x_shared2d(i1, i0)  X_shared[i1 * X_tile_width + i0] // size is X_tile_with * X_tile_width
    //
    //     for(int c = 0; c < C; c++){
    //
    //         if(h0 < K && w0 < K){
    //             k_shared2d(h0, w0) = k4d(m,c,h0,w0);
    //         }
    //         __syncthreads();
    //
    //         for (int i = h0; i < X_tile_width; i += TILE_WIDTH) {
    //             for (int j = w0; j < X_tile_width; j += TILE_WIDTH) {
    //                 x_shared2d(i,j) = x4d(b, c, i+h_base, j+w_base);
    //             }
    //         }
    //         __syncthreads();
    //
    //         for(int p = 0; p < K; ++p) {
    //             for (int q = 0; q < K; ++q) {
    //                 int pos_x = (h0+p) * X_tile_width + (w0+q);
    //                 int pos_k = p * K + q;
    //                 //acc += x_shared2d(h0+p, w0+q) * k_shared2d(p, q);
    //                 acc += X_shared[pos_x] * K_shared[pos_k];
    //             }
    //         }
    //         __syncthreads();
    //
    //     }
    //
    //     if (h < H_out && w < W_out){
    //         y4d(b,m,h,w) = acc;
    //     }
    //
    // #undef y4d
    // #undef x4d
    // #undef k4d
    // #undef k_shared2d
    // #undef x_shared2d
    // }

    // __global__ void unroll_kernel(int C, int H, int W, int K, int b, float* x, float* x_unroll){
    //   int c, s, h_out_index, w_out_index, h_unroll, w_unroll, h_base, p, q;
    //   int t = blockIdx.x * blockDim.x + threadIdx.x;
    //   // int H_out = H - K + 1;;
    //   // int W_out = W - K + 1;;
    //   int W_x_unroll = (H - K + 1) * (W - K + 1);
    //
    //   #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    //   #define x_unroll2d(i1, i0) x_unroll[(i1) * W_x_unroll + i0]
    //
    //   if (t < C * W_x_unroll) {
    //     c = t / W_x_unroll;
    //     s = t % W_x_unroll;
    //     h_out_index = s / (W - K + 1);
    //     w_out_index = s % (W - K + 1);
    //     w_unroll = s; //???
    //     // w_unroll = h_out_index * (W - K + 1) + h_out_index; // ???
    //     h_base = c*K*K;
    //     for (p = 0; p < K; p++) {
    //       for (q = 0; q < K; q++) {
    //         h_unroll = h_base + p*K + q; // ???
    //         x_unroll2d(h_unroll, w_unroll) = x4d(b, c, h_out_index+p, w_out_index+q);
    //       }
    //     }
    //   }
    //
    //   #undef x4d
    //   #undef x_unroll2d
    // }

    // void unroll(int C, int H, int W, int K, int b, float* x, float* x_unroll){
    //   int H_out = H - K + 1;
    //   int W_out = H - K + 1;
    //   int num_threads = C * H_out * W_out;
    //   dim3 blockDim1(MAX_NUM_THREADS, 1, 1);
    //   dim3 gridDim1(ceil((1.0*num_threads)/MAX_NUM_THREADS), 1, 1);
    //   unroll_kernel<<<gridDim1, blockDim1>>>(C, H, W, K, b, x.dptr_, x_unroll);
    // }

    // __global__ void matrixMultiply(float *A, float *B, float *C, int b, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
    //   //@@ Insert code to implement matrix multiplication here
    //   // Calculate the row index of the d_P element and d_M
    //   int Row = blockIdx.y*blockDim.y+threadIdx.y;
    //   // Calculate the column idenx of d_P and d_N
    //   int Col = blockIdx.x*blockDim.x+threadIdx.x;
    //
    //   if ((Row < numCRows) && (Col < numCColumns)) {
    //     float Pvalue = 0;
    //     // each thread computes one element of the block sub-matrix
    //     for (int k = 0; k < numAColumns; ++k) Pvalue += A[Row*numAColumns+k] * B[k*numBColumns+Col];
    //     C[b*numCRows*numCColumns + Row*numCColumns + Col] = Pvalue;
    //   }
    // }

    __global__ void matrix_multi_unroll_shared_kernal(float *k, float *x, float *y, int B, int M, int C, int H, int W, int K, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
      //@@ Insert code to implement matrix multiplication here
      //@@ You have to use shared memory for this MP
      __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
      __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
      __shared__ float T[TILE_WIDTH][TILE_WIDTH][TILE_WIDTH];

      int bx = blockIdx.x; int by = blockIdx.y;
      int tx = threadIdx.y; int ty = threadIdx.z;
      int b = blockIdx.z;
      int index = threadIdx.x;

      #define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
      #define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
      #define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

      int Row = by * TILE_WIDTH + ty;
      int Col = bx * TILE_WIDTH + tx;

      float Pvalue = 0.0;

      int W_out = W-K+1;
      int H_out = H-K+1;
      //

      // Loop over the M and N tiles required to compute the P element
      // The code assumes that the Width is a multiple of TILE_WIDTH!
      for (int i = 0; i < numAColumns/TILE_WIDTH + 1; ++i) {
         // Collaborative loading of M and N tiles into shared memory
        if ((Row < numARows) && (i*TILE_WIDTH+tx < numAColumns)) {
          int k_row = Row;
          int k_col = i*TILE_WIDTH + tx;
          int k_m = k_row;
          int k_c = k_col/(K*K);
          int k_h = k_col%(K*K)/K;
          int k_w = k_col%(K*K)%K;
          subTileM[ty][tx] = k4d(k_m, k_c, k_h, k_w);
        }
        else subTileM[ty][tx] = 0.0;
        __syncthreads();
        if ((i*TILE_WIDTH+ty < numBRows) && (Col < numBColumns) && (b < B)) {
          int x_row = i*TILE_WIDTH+ty;
          int x_col = Col;
          int x_b = b;
          int x_c = x_row/(K*K);
          int x_h = x_col/W_out;
          int x_w = x_col%W_out;
          int x_p = x_row%(K*K)/K;
          int x_q = x_row%(K*K)%K;
          subTileN[ty][tx] = x4d(x_b, x_c, x_h+x_p, x_w+x_q);
        }
        else subTileN[ty][tx] = 0.0;
        __syncthreads();

        // if ((Row < numCRows) && (Col < numCColumns)) {
            // for (int k=0; k<TILE_WIDTH; k++) {
            //     Pvalue += subTileM[ty][k] * subTileN[k][tx];
            // }
            //scan operation

            T[ty][tx][threadIdx.x] = subTileM[ty][threadIdx.x] * subTileN[threadIdx.x][tx];
            //__syncthreads();
            int stride = 1;
            while(stride < TILE_WIDTH){
                __syncthreads();
                int id = (threadIdx.x + 1) * stride * 2 - 1;
                if(id < TILE_WIDTH){
                    T[ty][tx][id] += T[ty][tx][id - stride];
                }
                stride = stride * 2;
            }
            stride = TILE_WIDTH / 2;
            while(stride > 0){
                __syncthreads();
                int id = (threadIdx.x + 1) * stride * 2 - 1;
                if((id + stride) < TILE_WIDTH){
                    T[ty][tx][id + stride] += T[ty][tx][id]; 
                }
                stride = stride / 2;
            }
            __syncthreads();
            Pvalue += T[ty][tx][TILE_WIDTH - 1];
            // __syncthreads();

            //koggle stome 
            // T[threadIdx.x] = subTileM[ty][threadIdx.x] * subTileN[threadIdx.x][tx];
            // for(int stride = TILE_WIDTH ; stride >= 1; stride = stride >> 1){
            //     __syncthreads();
            //     if(threadIdx.x < stride) T[threadIdx.x] += T[threadIdx.x + stride];
            // }
            // __syncthreads();
            // Pvalue += T[0];
            // __syncthreads();
            // for (int k=0; k<TILE_WIDTH; k++) {
            //     Pvalue += subTileM[ty][k] * subTileN[k][tx];
            // }
            // if((Row < numCRows) && (Col < numCColumns) && threadIdx.x == 0)
                

            
        // }
      }
      __syncthreads();

      if ((Row < numCRows) && (Col < numCColumns) && (b < B)) {
        int y_m = Row;
        int y_h = Col/W_out;
        int y_w = Col%W_out;
        y4d(b, y_m, y_h, y_w) = Pvalue; //
      }

      #undef y4d
      #undef x4d
      #undef k4d
    }

    // void matrix_multi_shared(float *A, float *B, float *C, int b, int numARows, int numAColumns,
    //                                                               int numBRows, int numBColumns,
    //                                                               int numCRows, int numCColumns){
    //   dim3 dimGrid2(ceil((1.0*numCColumns)/TILE_WIDTH), ceil((1.0*numCRows)/TILE_WIDTH), 1);
    //   dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
    //   matrix_multi_shared_kernal<<<dimGrid2, dimBlock2>>>(A, B, C, b, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
    // }

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

        // Use mxnet's CHECK_EQ to do assertions.
        // Remove this assertion when you do your implementation!

        // Extract the tensor dimensions into B,M,C,H,W,K
        const int B = x.shape_[0];
        const int M = y.shape_[1];
        const int C = x.shape_[1];
        const int H = x.shape_[2];
        const int W = x.shape_[3];
        const int K = k.shape_[3];
        const int H_out = H-K+1;
        const int W_out = W-K+1;
        // const int H_unroll = H_out*W_out;
        // const int W_unroll = C*K*K;
        const int numARows = M;    // number of rows in the matrix A
        const int numAColumns = C*K*K; // number of columns in the matrix A
        const int numBRows = C*K*K;    // number of rows in the matrix B
        const int numBColumns = W_out*H_out; // number of columns in the matrix B
        const int numCRows = numARows;    // number of rows in the matrix C (you have to set this)
        const int numCColumns = numBColumns;
        // float* device_x_unroll;
        // cudaMalloc((void**)&device_x_unroll, W_unroll*H_unroll*sizeof(float));

        // for (int b = 0; b < B; b++) {
        //   // unroll(C, H, W, K, b, x, device_X_unroll);
        //   int H_out = H - K + 1;
        //   int W_out = H - K + 1;
        //   int num_threads = C * H_out * W_out;
        //   dim3 blockDim1(MAX_NUM_THREADS, 1, 1);
        //   dim3 gridDim1(ceil((1.0*num_threads)/MAX_NUM_THREADS), 1, 1);
        //   unroll_kernel<<<gridDim1, blockDim1>>>(C, H, W, K, b, x.dptr_, device_x_unroll);
        //
        //   // matrix_multi_shared(k, device_x_unroll, y, b, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
        //   dim3 dimGrid2(ceil((1.0*numCColumns)/TILE_WIDTH), ceil((1.0*numCRows)/TILE_WIDTH), B);
        //   dim3 dimBlock2(TILE_WIDTH, TILE_WIDTH, 1);
        //   matrixMultiply<<<dimGrid2, dimBlock2>>>(k.dptr_, x.dptr_, y.dptr_, B, M, C, H, W, K numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
        //   // matrix_multi_shared_kernal<<<dimGrid2, dimBlock2>>>(k.dptr_, device_x_unroll, y.dptr_, b, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
        // }

        dim3 dimGrid(ceil((1.0*numCColumns)/TILE_WIDTH), ceil((1.0*numCRows)/TILE_WIDTH), B);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
        matrix_multi_unroll_shared_kernal<<<dimGrid, dimBlock>>>(k.dptr_, x.dptr_, y.dptr_, B, M, C, H, W, K, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

        // // Set the kernel dimensions
        // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
        // int W_grid = ceil(float(W_out)/TILE_WIDTH);
        // int H_grid = ceil(float(H_out)/TILE_WIDTH);
        // int Z = W_grid * H_grid;
        // dim3 gridDim(B, M, Z);
        // size_t shmem_size = sizeof(float) * ((TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) + K * K);
        //
        // // Call the kernel
        // forward_kernel_optimized<<<gridDim, blockDim, shmem_size>>>(y.dptr_,x.dptr_,k.dptr_,B,M,C,H,W,K,W_grid);

        // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
        MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
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
