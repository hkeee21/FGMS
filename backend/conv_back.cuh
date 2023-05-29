#include <cuda_runtime.h>
#include <cuda.h>
#include <mma.h>
#include <cuda/pipeline>

/*******************************************************************
device functions
*/
__device__ __forceinline__ int binary_search_back(
                            const int *S_csrRowPtr, const int eid, 
                            const int start, const int end) {
    
    int lo = start, hi = end;
    if (lo == hi){
        return lo;
    }
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(S_csrRowPtr + mid) <= eid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (__ldg(S_csrRowPtr + hi) <= eid) {
        return hi;
    } else {
        return hi - 1;
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
__global__ void _fgms_fusion_tf32_W_transpose(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ out_f_grad, 
                const float *__restrict__ kw, 
                float *in_f_grad,
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
  const int warp_row = warpId / NS; // 0, 1, 2, 3
  const int warp_col = warpId % NS; // 0, 1

  // Weight index
  const int widx = binary_search_back(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
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
  for (int s = 0; s < c_out; s += BLOCK_SIZE) {
    // Load the matrices from device memory
    // to shared memory; each thread loads
    // one element of each matrix

    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = (ty < c_in && (s + ctx) < c_out) ? 
      *((float4*)(kw_ptr + c_out * ty + s + ctx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;

      // The thread deals with the x-th channel of the y-th output
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_out && in_row > -1) ? 
        *((float4*)(&out_f_grad[c_out * in_row + s + ctx])) : 
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
      wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::col_major> b;
      // wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
      wmma::load_matrix_sync(b, &Bs[warp_col * N][k], BLOCK_SIZE + SKEW);
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
    if (out_row > -1 && cx < c_in){
#pragma unroll
      for (int c = 0; c < 4; c++){
        atomicAdd(&in_f_grad[c_in * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, KLOOP = 4, SKEW = 8, 
M = 16, K = 8, N = 16, 
MS = BLOCK_SIZE / M = 2, 
NS = BLOCK_SIZE / N = 2, 
KS = warpNum / MS / NS = 1,
blockDim.x = 8, blockDim.y = 16
*/
template <int BLOCK_SIZE, int KLOOP, int SKEW, 
  int M, int K, int N, int MS, int NS, int KS>
__global__ void _fgms_fusion_tf32_I_transpose(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const float *__restrict__ in_f, 
                const float *__restrict__ out_g, 
                float *w_g,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 800
  // Block index
  const int bx = blockIdx.x;
  // const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 2;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpNum = blockDim.x * blockDim.y / 32;
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS;
  const int warp_col = warpId % NS;

  // Weight index
  const int widx = binary_search(qkpos, bx * KLOOP * (BLOCK_SIZE / 2), 0, k_vol);
  float *wg_ptr = &w_g[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = ctx;
  const int y = (BLOCK_SIZE / 2) * KLOOP * bx + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  float padding[4] = {0.0f};

  // Declaration of the shared memory array As and Bs
  // TODO: wonder if KLOOP can be reduced
  // TODO: BLOCK_SIZE of different dim can be different
  __shared__ float As[KLOOP][(BLOCK_SIZE / 2)][BLOCK_SIZE + SKEW];
  __shared__ float Bs[KLOOP][(BLOCK_SIZE / 2)][BLOCK_SIZE + SKEW];
  __shared__ float Cs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, float> c;
  wmma::fragment<wmma::matrix_a, M, N, K, wmma::precision::tf32, wmma::col_major> a;
  wmma::fragment<wmma::matrix_b, M, N, K, wmma::precision::tf32, wmma::row_major> b;

#pragma unroll
  for (int m = 0; m < c_in; m += BLOCK_SIZE){
#pragma unroll
    for (int n = 0; n < c_out; n += BLOCK_SIZE){
        // empty the accumulation space
        wmma::fill_fragment(c, 0.0f);
#pragma unroll
      for (int k = 0; k < KLOOP; k++){
        int y_temp = y + k * (BLOCK_SIZE / 2);
        int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;
        int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;

        *((float4*)(&As[k][ty][ctx])) = ((m + ctx) < c_in && in_row > -1) ? 
          *((float4*)(&in_f[c_in * in_row + m + ctx])) : 
          *((float4*)(&padding[0]));

        *((float4*)(&Bs[k][ty][ctx])) = ((n + ctx) < c_out && out_row > -1) ? 
          *((float4*)(&out_g[c_out * out_row + n + ctx])) : 
          *((float4*)(&padding[0]));
      }
      __syncthreads();
#pragma unroll     
      for (int k = 0; k < KLOOP * BLOCK_SIZE / 2; k += K){
        int i = k / (BLOCK_SIZE / 2);
        int j = k % (BLOCK_SIZE / 2);
        wmma::load_matrix_sync(a, &As[i][j][warp_row * M], BLOCK_SIZE + SKEW);
        wmma::load_matrix_sync(b, &Bs[i][j][warp_col * N], BLOCK_SIZE + SKEW);
#pragma unroll
        for (int t = 0; t < a.num_elements; t++) {
          a.x[t] = wmma::__float_to_tf32(a.x[t]);
        }
#pragma unroll
        for (int t = 0; t < b.num_elements; t++) {
          b.x[t] = wmma::__float_to_tf32(b.x[t]);
        }
        wmma::mma_sync(c, a, b, c); 
      }
      wmma::store_matrix_sync(&Cs[warp_row * M][warp_col * N], 
        c, BLOCK_SIZE + SKEW, wmma::mem_row_major);
      // make sure all the partial sums are stored into shared memory
      __syncthreads();
#pragma unroll
      for (int y = 0; y < 2; y++){
        for (int c = 0; c < 4; c++){
          atomicAdd(wg_ptr + (m + y * M + ty) * c_out + (n + cx) + c, 
            Cs[y * M + ty][ctx + c]);
        }
      }
    }
  }
#endif
}

