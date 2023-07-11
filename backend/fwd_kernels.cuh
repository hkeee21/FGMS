#include <mma.h>
#include <cuda/pipeline>
#include "utils.cuh"

using namespace nvcuda;

/*
BLOCK_SIZE = 32,   
EX_LOOP = 1, IM_LOOP = 4, 
SKEW = 8, 
M = 16, K = 16, N = 16, 
MS = IM_LOOP * BLOCK_SIZE / M = 8, 
NS = BLOCK_SIZE / N = 2, 
KS = warpNum / MS / NS = 1, 
blockDim.x = 4, blockDim.y = 128
*/
template <int BLOCK_SIZE, int EX_LOOP, int IM_LOOP, 
  int SKEW, int M, int K, int N, int MS, int NS, int KS>
__global__ void _fgms_fusion_fp16_v2(
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
  const int ctx_a = tx << 3;
  const int ctx_b = tx << 1;  // 1 if MS == 8, 2 if MS == 4
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS; // 0, 1, 2, 3, 4, 5, 6, 7
  const int warp_col = warpId % NS; // 0, 1

  // Weight index
  const int widx = binary_search(
    qkpos, by * EX_LOOP * IM_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx_a = BLOCK_SIZE * bx + ctx_a;
  const int cx_b = BLOCK_SIZE * bx + ctx_b;
  const int y = BLOCK_SIZE * EX_LOOP * IM_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array
  __shared__ half As[EX_LOOP][IM_LOOP * BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[EX_LOOP];

#pragma unroll
  for (int _n = 0; _n < EX_LOOP; _n++){
    wmma::fill_fragment(c[_n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
#pragma unroll
  for (int _k = 0; _k < c_in; _k += BLOCK_SIZE) {
    int b_load_x = ty / 4;
    int b_load_y_local = ty % 4 * 8 + ctx_b;
    int b_load_y_global = ty % 4 * 8 + cx_b;
    // Kernel weight to Bs
    *((half2*)(&Bs[b_load_x][b_load_y_local])) = 
      ((_k + b_load_x) < c_in && b_load_y_global < c_out) ? 
      *((half2*)(kw_ptr + c_out * (b_load_x + _k) + b_load_y_global)) : 
      *((half2*)(&padding[0]));
    
    // Input feature to As
#pragma unroll
    for (int _n = 0; _n < EX_LOOP; _n++){

      int y_temp = y + _n * IM_LOOP * BLOCK_SIZE;
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[_n][ty][ctx_a])) = ((_k + ctx_a) < c_in && in_row > -1) ? 
        *((float4*)(&in_f[c_in * in_row + _k + ctx_a])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[EX_LOOP];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;
      wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < EX_LOOP; n++){
        wmma::load_matrix_sync(a[n], &As[n][warp_row * M][k], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c[n], a[n], b, c[n]); 
      }  
    }
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < EX_LOOP; n++){
    wmma::store_matrix_sync(&As[n][warp_row * M][warp_col * N], 
      c[n], BLOCK_SIZE + SKEW, wmma::mem_row_major);
  }

  // Synchronize to make sure that all C fragments are 
  // stored into shared memory
  __syncthreads();

  // Write the block sub-matrix to device memory;
  // each thread writes one element
#pragma unroll
  for (int n = 0; n < EX_LOOP; n++){
    int y_temp = y + n * IM_LOOP * BLOCK_SIZE;
    int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;
    if (out_row > -1 && cx_a < c_out){
#pragma unroll
      for (int _c = 0; _c < 8; _c += 2){
        atomicAdd(((half2*)(&out_f[c_out * out_row + cx_a + _c])), 
          *((half2*)(&As[n][ty][ctx_a + _c])));
      }
    }
  }
#endif
}