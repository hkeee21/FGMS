#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cuda/pipeline>
#include "utils.cuh"

#define DIV_UP(x, y) ((x) + (y) - 1) / (y)
#define _FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define _FLOAT2(pointer) (reinterpret_cast<float2 *>(&(pointer))[0])
#define _HALF2(pointer) (reinterpret_cast<half2 *>(&(pointer))[0])

/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, 
blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp32_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][2] = {0.0f};
  float padding[2] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, 
blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp32_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP] = {0.0f};
  float padding = 0.0f;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // float Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
        Csub[n] += As[n][ty][k] * Bs[k][tx];
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, 
blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_seq_fp32_1(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx;

  // Weight index
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP] = {0.0f};
  float padding = 0.0f;

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // float Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
        Csub[n] += As[n][ty][k] * Bs[k][tx];
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      out_f[c_out * out_row + cx] += Csub[n];
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_seq_fp32(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const float *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        out_f[c_out * out_row + cx + c] += Csub[n][c];
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        float Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] += Ast * Bs[k][ctx + c];
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 4, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp32_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Load the matrices from device memory
  // to shared memory; each thread loads
  // one element of each matrix

  // Kernel weight to Bs
  *((float4*)(&Bs[ty][ctx])) = ((ty) < c_in && cx < c_out) ? 
    *((float4*)(kw_ptr + c_out * (ty) + cx)) : 
    *((float4*)(&padding[0]));
    
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    int y_temp = y + n * BLOCK_SIZE;

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    *((float4*)(&As[n][ty][ctx])) = ((ctx) < c_in && in_row > -1) ? 
      *((float4*)(&in_f[c_in * in_row + ctx])) : 
      *((float4*)(&padding[0]));
  }

  // Synchronize to make sure the matrices are loaded
  __syncthreads();

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll 
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k) {
      float Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] += Ast * Bs[k][ctx + c];
      }
    }
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 4, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_8(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 3;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][8] = {__float2half(0.0f)};
  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 8; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 8; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_4(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][4] = {__float2half(0.0f)};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    int y_temp = y;
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
      
      y_temp += BLOCK_SIZE;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k){
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 4; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_4_once(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][4] = {__float2half(0.0f)};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  // In "loop once" version, s = 0
  // for (int s = 0; s < c_in; s += BLOCK_SIZE) {

  // Kernel weight to Bs
  *((float2*)(&Bs[ty][ctx])) = (ty < c_in && cx < c_out) ? 
    *((float2*)(kw_ptr + c_out * ty + cx)) : 
    *((float2*)(&padding[0]));
    
  int y_temp = y;
  // Input feature to As
  for (int n = 0; n < N_LOOP; n++){

    // The thread deals with the x-th channel of the y-th output
    int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

    *((float2*)(&As[n][ty][ctx])) = (ctx < c_in && in_row > -1) ? 
      *((float2*)(&in_f[c_in * in_row + ctx])) : 
      *((float2*)(&padding[0]));
      
    y_temp += BLOCK_SIZE;
  }

  // Synchronize to make sure the matrices are loaded
  __syncthreads();

  // Multiply the two matrices together;
  // each thread computes one element
  // of the block sub-matrix
#pragma unroll 
  for (int n = 0; n < N_LOOP; n++){
#pragma unroll
    for (int k = 0; k < c_in; ++k){
      half Ast = As[n][ty][k];
#pragma unroll
      for (int c = 0; c < 4; c++){
        Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
      }
    }

    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 8, SKEW = 8, blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP][2] = {__float2half(0.0f)};
  half padding[2] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        half Ast = As[n][ty][k];
#pragma unroll
        for (int c = 0; c < 2; c++){
          Csub[n][c] = __hfma(Ast, Bs[k][ctx + c], Csub[n][c]);
        }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 2; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], Csub[n][c]);
      }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_fusion_fp16_1(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx << 1;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP] = {__float2half(0.0f)};
  half padding = __float2half(0.0f);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        // half Ast = As[n][ty][k];
        // for (int c = 0; c < 2; c++){
          Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
        // }
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      // for (int c = 0; c < 2; c++){
      atomicAdd(&out_f[c_out * out_row + cx], Csub[n]);
      // }
    }
  }
}


/*
BLOCK_SIZE = 16, N_LOOP = 4, SKEW = 8, blockDim.x = 16, blockDim.y = 16
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW>
__global__ void _fgms_seq_fp16_1(
                const int knnz,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  // const int ctx = tx << 1;

  // Weight index
  const half *kw_ptr = &kw[0];

  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + tx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  half Csub[N_LOOP] = {__float2half(0.0f)};
  half padding = __float2half(0.0f);

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  
  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {

    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    Bs[ty][tx] = ((s + ty) < c_in && cx < c_out) ? 
      *(kw_ptr + c_out * (s + ty) + cx) : 
      padding;
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      As[n][ty][tx] = ((s + tx) < c_in && in_row > -1) ? 
        in_f[c_in * in_row + s + tx] : 
        padding;
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together;
    // each thread computes one element
    // of the block sub-matrix
#pragma unroll 
    for (int n = 0; n < N_LOOP; n++){
#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k) {
        Csub[n] = __hfma(As[n][ty][k], Bs[k][tx], Csub[n]);
      }
    }

    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
      out_f[c_out * out_row + cx] = 
        __hadd(out_f[c_out * out_row + cx], Csub[n]);
    }
  }
}


using namespace nvcuda;
/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_tf32(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 800
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const float *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = wmma::__float_to_tf32(a[n].x[t]);
        }
        wmma::mma_sync(c[n], a[n], b, c[n]); 
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 8, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_seq_tf32(
                const int knnz,
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ kw, 
                float *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 800
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const float *kw_ptr = &kw[0];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  float padding[4] = {0.0f};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float4*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
      }
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a[n].num_elements; t++) {
          a[n].x[t] = wmma::__float_to_tf32(a[n].x[t]);
        }
        wmma::mma_sync(c[n], a[n], b, c[n]); 
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        out_f[c_out * out_row + cx + c] += As[n][ty][ctx + c];
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_fp16_tc4(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
    // wmma::fill_fragment(c[n], 0.0f);
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_seq_fp16(
                const int knnz,
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  // const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  // const half *kw_ptr = &kw[widx * c_in * c_out];
  const half *kw_ptr = &kw[0];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty;

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float2*)(&Bs[ty][ctx])) = ((s + ty) < c_in && cx < c_out) ? 
      *((float2*)(kw_ptr + c_out * (s + ty) + cx)) : 
      *((float2*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < knnz ? imap[y_temp] : -1;

      *((float2*)(&As[n][ty][ctx])) = ((s + ctx) < c_in && in_row > -1) ? 
        *((float2*)(&in_f[c_in * in_row + s + ctx])) : 
        *((float2*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }  
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < knnz ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        // out_f[c_out * out_row + cx + c] += As[n][ty][ctx + c];
        out_f[c_out * out_row + cx + c] = 
          __hadd(out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 8, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_fp16_tc4_async(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ kw, 
                half *out_f,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  // const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  // Csub is used to store the element of the block sub-matrix
  // that is computed by the thread
  // float Csub[N_LOOP][4] = {0.0f};
  half padding[4] = {__float2half(0.0f)};

  // Declaration of the shared memory array As used to
  // store the sub-matrix of A
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memory array Bs used to
  // store the sub-matrix of B
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Declaration of the shared memeory array Cs used to
  // store the sub-matrix of C
  // __shared__ float Cs[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Pipelined copy between gmem and shmem
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape4 = cuda::aligned_size_t<alignof(float2)>(sizeof(float2));

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP / 2];

#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  // required to compute the block sub-matrix
  for (int s = 0; s < c_in; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    // const half *kw2Bs_ptr = ((s + ty) < c_in && cx < c_out) ? 
    //   kw_ptr + c_out * (s + ty) + cx : &padding[0];
    pipe.producer_acquire();
    if ((s + ty) < c_in && cx < c_out){
      cuda::memcpy_async(&Bs[ty][ctx], kw_ptr + c_out * (s + ty) + cx, shape4, pipe);
    }
    else{
      cuda::memcpy_async(&Bs[ty][ctx], &padding[0], shape4, pipe);
    }
    // cuda::memcpy_async(&Bs[ty][ctx], kw2Bs_ptr, shape4, pipe);
    pipe.producer_commit();
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      // const half *inf2As_ptr = ((s + ctx) < c_in && in_row > -1) ? 
      //   &in_f[c_in * in_row + s + ctx] : &padding[0];
      pipe.producer_acquire();
      if ((s + ctx) < c_in && in_row > -1){
        cuda::memcpy_async(&As[n][ty][ctx], &in_f[c_in * in_row + s + ctx], shape4, pipe);
      }
      else{
        cuda::memcpy_async(&As[n][ty][ctx], &padding[0], shape4, pipe);
      }
      // cuda::memcpy_async(&As[n][ty][ctx], inf2As_ptr, shape4, pipe);
      pipe.producer_commit();
    }

    // Synchronize to make sure the matrices are loaded
    cuda::pipeline_consumer_wait_prior<0>(pipe);
    __syncthreads();

    // Multiply the two matrices together using Tensor Core
    // Load data from shmem to tensor core
    // Just load Bs once
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP / 2];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::load_matrix_sync(a[n], &As[n * MS + warpId / WS][warp_row % MS * M][k], BLOCK_SIZE + SKEW);
      }  
#pragma unroll 
      for (int n = 0; n < N_LOOP / 2; n++){
        wmma::mma_sync(c[n], a[n], b, c[n]);
      }
    }
    // Synchronize to make sure that the preceding
    // computation is done before loading two new
    // sub-matrices of A and B in the next iteration
    pipe.consumer_release();
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP / 2; n++){
    wmma::store_matrix_sync(&As[n * MS + warpId / WS][warp_row % MS * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    int y_temp = y + n * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx < c_out){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&out_f[c_out * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}
