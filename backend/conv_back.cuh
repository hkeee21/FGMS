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

  float padding[4] = {0.0f};

  // Declaration of the shared memory array
  __shared__ float As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
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
  for (int s = 0; s < c_out; s += BLOCK_SIZE) {
    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = (ty < c_in && (s + ctx) < c_out) ? 
      *((float4*)(kw_ptr + c_out * ty + s + ctx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_out && in_row > -1) ? 
        *((float4*)(&out_f_grad[c_out * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

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
    // CUDA_ARCH >= 9.0
    //   atomicAdd(((float4*)(&in_f_grad[c_in * out_row + ctx])), 
    //     *((float4*)(&As[n][ty][ctx])));
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, N_LOOP = 4, SKEW = 8, M = 16, K = 16, N = 16, 
MS = 2, NS = 2, WS = 4 = MS x NS
blockDim.x = 4, blockDim.y = 32
*/
template <int BLOCK_SIZE, int N_LOOP, int SKEW, 
  int M, int K, int N, int WS, int MS, int NS>
__global__ void _fgms_fusion_fp16_W_transpose(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ out_f_grad, 
                const half *__restrict__ kw, 
                half *in_f_grad,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 3;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS; // 0, 1
  const int warp_col = warpId % NS; // 0, 1

  // Weight index
  const int widx = binary_search_back(qkpos, by * N_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx;
  const int y = BLOCK_SIZE * N_LOOP * by + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array
  __shared__ half As[N_LOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c[N_LOOP];

#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    wmma::fill_fragment(c[n], __float2half(0.0f));
  }
  
  // May not be necessary
  __syncthreads();

  // Loop over all the sub-matrices of A and B
  for (int s = 0; s < c_out; s += BLOCK_SIZE) {
    // Kernel weight to Bs
    *((float4*)(&Bs[ty][ctx])) = (ty < c_in && (s + ctx) < c_out) ? 
      *((float4*)(kw_ptr + c_out * ty + s + ctx)) : 
      *((float4*)(&padding[0]));
    
    // Input feature to As
    for (int n = 0; n < N_LOOP; n++){

      int y_temp = y + n * BLOCK_SIZE;
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[n][ty][ctx])) = ((s + ctx) < c_out && in_row > -1) ? 
        *((float4*)(&out_f_grad[c_out * in_row + s + ctx])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[N_LOOP];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b;
      // wmma::load_matrix_sync(b, &Bs[k][warp_col * N], BLOCK_SIZE + SKEW);
      wmma::load_matrix_sync(b, &Bs[warp_col * N][k], BLOCK_SIZE + SKEW);
#pragma unroll
      for (int n = 0; n < N_LOOP; n++){
        wmma::load_matrix_sync(a[n], &As[n][warp_row * M][k], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c[n], a[n], b, c[n]); 
      }  
    }
    __syncthreads();
  }

  // Store C fragments to shared memory
  // Note that we reuse As for Cs storing
#pragma unroll
  for (int n = 0; n < N_LOOP; n++){
    wmma::store_matrix_sync(&As[n][warp_row * M][warp_col * N], 
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
      for (int c = 0; c < 8; c++){
        atomicAdd(&in_f_grad[c_in * out_row + cx + c], As[n][ty][ctx + c]);
      }
    }
  }
#endif
}


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
__global__ void _fgms_fusion_fp16_W_transpose_v2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ out_f_grad, 
                const half *__restrict__ kw, 
                half *in_f_grad,
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
  const int widx = binary_search_back(
    qkpos, by * EX_LOOP * IM_LOOP * BLOCK_SIZE, 0, k_vol);
  const half *kw_ptr = &kw[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = BLOCK_SIZE * bx + ctx_a;
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
  for (int _k = 0; _k < c_out; _k += BLOCK_SIZE) {
    int b_load_x = ty / 4;
    int b_load_y = ty % 4 * 8 + ctx_b;
    // Kernel weight to Bs
    *((half2*)(&Bs[b_load_x][b_load_y])) = 
      (b_load_x < c_in && (_k + b_load_y) < c_out) ? 
      *((half2*)(kw_ptr + c_out * b_load_x + _k + b_load_y)) : 
      *((half2*)(&padding[0]));
    
    // Input feature to As
    for (int _n = 0; _n < EX_LOOP; _n++){

      int y_temp = y + _n * IM_LOOP * BLOCK_SIZE;
      int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;

      *((float4*)(&As[_n][ty][ctx_a])) = ((_k + ctx_a) < c_out && in_row > -1) ? 
        *((float4*)(&out_f_grad[c_out * in_row + _k + ctx_a])) : 
        *((float4*)(&padding[0]));
    }

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k += K){
      wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[EX_LOOP];
      wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b;
      wmma::load_matrix_sync(b, &Bs[warp_col * N][k], BLOCK_SIZE + SKEW);
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
    if (out_row > -1 && cx < c_in){
#pragma unroll
      for (int _c = 0; _c < 8; _c += 2){
        atomicAdd(((half2*)(&in_f_grad[c_in * out_row + cx + _c])), 
          *((half2*)(&As[n][ty][ctx_a + _c])));
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
// CUDA_ARCH >= 9.0
//           atomicAdd(((float4*)(wg_ptr + (m + y * M + ty) * c_out + (n + cx))),
//             *((float4*)(&Cs[y * M + ty][ctx])));
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, KLOOP = 8, SKEW = 8, 
M = 16, K = 16, N = 16, 
MS = BLOCK_SIZE / M = 2,
NS = BLOCK_SIZE / N = 2, 
KS = warpNum / MS / NS = 1,
blockDim.x = 4, blockDim.y = 32
*/
template <int BLOCK_SIZE, int KLOOP, int SKEW, 
  int M, int K, int N, int MS, int NS, int KS>
__global__ void _fgms_fusion_fp16_I_transpose(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ out_g, 
                half *w_g,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  // const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 3;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpNum = blockDim.x * blockDim.y / 32;
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS; // 0, 1
  const int warp_col = warpId % NS; // 0, 1

  // Weight index
  const int widx = binary_search(qkpos, bx * KLOOP * BLOCK_SIZE, 0, k_vol);
  half *wg_ptr = &w_g[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = ctx;
  const int y = BLOCK_SIZE * KLOOP * bx + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array As and Bs
  // TODO: wonder if KLOOP can be reduced
  // TODO: BLOCK_SIZE of different dim can be different
  __shared__ half As[KLOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ half Bs[KLOOP][BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ half Cs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> a;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;

#pragma unroll
  for (int m = 0; m < c_in; m += BLOCK_SIZE){
#pragma unroll
    for (int n = 0; n < c_out; n += BLOCK_SIZE){
        // empty the accumulation space
        wmma::fill_fragment(c, __float2half(0.0f));
#pragma unroll
      for (int k = 0; k < KLOOP; k++){
        int y_temp = y + k * BLOCK_SIZE;
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
      for (int k = 0; k < KLOOP * BLOCK_SIZE; k += K){
        int i = k / BLOCK_SIZE;
        int j = k % BLOCK_SIZE;
        wmma::load_matrix_sync(a, &As[i][j][warp_row * M], BLOCK_SIZE + SKEW);
        wmma::load_matrix_sync(b, &Bs[i][j][warp_col * N], BLOCK_SIZE + SKEW);
        wmma::mma_sync(c, a, b, c); 
      }
      wmma::store_matrix_sync(&Cs[warp_row * M][warp_col * N], 
        c, BLOCK_SIZE + SKEW, wmma::mem_row_major);
      // make sure all the partial sums are stored into shared memory
      __syncthreads();
#pragma unroll
      for (int c = 0; c < 8; c++){
        atomicAdd(wg_ptr + (m + ty) * c_out + (n + cx) + c, 
          Cs[ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE = 32, KLOOP = 4, SKEW = 8, 
M = 16, K = 16, N = 16, 
MS = BLOCK_SIZE / M = 2,
NS = BLOCK_SIZE / N = 2, 
KS = BLOCK_SIZE / K = 2,
blockDim.x = 4, blockDim.y = 32
*/
template <int BLOCK_SIZE, int KLOOP, int SKEW, 
  int M, int K, int N, int MS, int NS, int KS>
__global__ void _fgms_fusion_fp16_I_transpose_v2(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ out_g, 
                half *w_g,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int GridY = gridDim.y;
  const int bz = blockIdx.z;
  const int GridZ = gridDim.z;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 3;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpNum = blockDim.x * blockDim.y / 32;
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_row = warpId / NS; // 0, 1
  const int warp_col = warpId % NS; // 0, 1

  // Weight index
  const int widx = binary_search(qkpos, bx * KLOOP * BLOCK_SIZE, 0, k_vol);
  half *wg_ptr = &w_g[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  const int cx = ctx;
  const int y = BLOCK_SIZE * KLOOP * bx + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array As and Bs
  // TODO: wonder if KLOOP can be reduced
  // TODO: BLOCK_SIZE of different dim can be different
  __shared__ half As[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  __shared__ half Bs[BLOCK_SIZE][BLOCK_SIZE + SKEW];
  // __shared__ half Cs[BLOCK_SIZE][BLOCK_SIZE + SKEW];

#pragma unroll
  for (int mi = 0; mi < c_in; mi += GridY * BLOCK_SIZE){
    int m = mi + by * BLOCK_SIZE;
    if (m >= c_in) break;
#pragma unroll
    for (int ni = 0; ni < c_out; ni += GridZ * BLOCK_SIZE){
      int n = ni + bz * BLOCK_SIZE;
      if (n >= c_out) break;
      // Fragments to store As, Bs and Cs
      wmma::fragment<wmma::accumulator, M, N, K, half> c;
      // empty the accumulation space
      wmma::fill_fragment(c, __float2half(0.0f));
#pragma unroll
      for (int k = 0; k < KLOOP; k++){

        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> a;
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;

        int y_temp = y + k * BLOCK_SIZE;
        int in_row = y_temp < __ldg(&kpos[widx + 1]) ? imap[y_temp] : -1;
        int out_row = y_temp < __ldg(&kpos[widx + 1]) ? omap[y_temp] : -1;

        *((float4*)(&As[ty][ctx])) = ((m + ctx) < c_in && in_row > -1) ? 
          *((float4*)(&in_f[c_in * in_row + m + ctx])) : 
          *((float4*)(&padding[0]));

        *((float4*)(&Bs[ty][ctx])) = ((n + ctx) < c_out && out_row > -1) ? 
          *((float4*)(&out_g[c_out * out_row + n + ctx])) : 
          *((float4*)(&padding[0]));
        
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < KS; ki++){
          wmma::load_matrix_sync(a, &As[ki * K][warp_row * M], BLOCK_SIZE + SKEW);
          wmma::load_matrix_sync(b, &Bs[ki * K][warp_col * N], BLOCK_SIZE + SKEW);
          wmma::mma_sync(c, a, b, c); 
        }
        __syncthreads();
      }
      wmma::store_matrix_sync(&As[warp_row * M][warp_col * N], 
        c, BLOCK_SIZE + SKEW, wmma::mem_row_major);
      // make sure all the partial sums are stored into shared memory
      __syncthreads();
#pragma unroll
      for (int c = 0; c < 8; c++){
        atomicAdd(wg_ptr + (m + ty) * c_out + (n + cx) + c, 
          As[ty][ctx + c]);
      }
    }
  }
#endif
}


/*
BLOCK_SIZE_X = 32, 
BLOCK_SIZE_Y = 128, 
KLOOP = 1, SKEW = 8, 
M = 16, K = 16, N = 16, 
MS = BLOCK_SIZE_X / M = 2, 
NS = BLOCK_SIZE_X / N = 2, 
KS = warpNum / MS / NS = 4,
blockDim.x = 4, blockDim.y = 128
*/
template <int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int KLOOP, int SKEW, 
  int M, int K, int N, int MS, int NS, int KS>
__global__ void _fgms_fusion_fp16_I_transpose_v3(
                const int *__restrict__ kpos,
                const int *__restrict__ qkpos, 
                const int k_vol, 
                const int c_in,
                const int c_out,
                const half *__restrict__ in_f, 
                const half *__restrict__ out_g, 
                half *w_g,
                const int *imap, 
                const int *omap) {
#if __CUDA_ARCH__ >= 700
  // Block index
  const int bx = blockIdx.x;
  // const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int ctx = tx << 3;
  const int sx = ty % 4 * 8 + tx * 2;
  const int sy = ty / 4;
  // const int ctx2 = tx << 1;
  const int tid = ty * blockDim.x + tx;

  // Warp index
  const int warpNum = blockDim.x * blockDim.y / 32;
  const int warpId = tid / 32;
  const int laneId = tid % 32;
  const int warp_k = warpId / (MS * NS);  // 0, 1, 2, 3
  const int warp_m = warpId % (MS * NS) / NS; // 0, 1
  const int warp_n = warpId % (MS * NS) % NS; // 0, 1

  // Weight index
  const int widx = binary_search(qkpos, bx * KLOOP * BLOCK_SIZE_Y, 0, k_vol);
  half *wg_ptr = &w_g[widx * c_in * c_out];
  
  // Coordinate. x is for rows, y is for columns.
  // const int cx = ctx;
  const int y = BLOCK_SIZE_Y * KLOOP * bx + ty 
    - __ldg(&qkpos[widx]) + __ldg(&kpos[widx]);

  half padding[8] = {__float2half(0.0f)};

  // Declaration of the shared memory array As and Bs
  // TODO: wonder if KLOOP can be reduced
  // TODO: BLOCK_SIZE of different dim can be different
  __shared__ half As[KLOOP][BLOCK_SIZE_Y][BLOCK_SIZE_X + SKEW];
  __shared__ half Bs[KLOOP][BLOCK_SIZE_Y][BLOCK_SIZE_X + SKEW];
  __shared__ half Cs[KS][BLOCK_SIZE_X][BLOCK_SIZE_X + SKEW];

  // Fragments to store As, Bs and Cs
  wmma::fragment<wmma::accumulator, M, N, K, half> c;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> a;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b;

#pragma unroll
  for (int m = 0; m < c_in; m += BLOCK_SIZE_X){
#pragma unroll
    for (int n = 0; n < c_out; n += BLOCK_SIZE_X){
      // empty the accumulation space
      wmma::fill_fragment(c, __float2half(0.0f));
#pragma unroll
      for (int k = 0; k < KLOOP; k++){
        int y_temp = y + k * BLOCK_SIZE_Y;
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
      for (int _k = 0; _k < KLOOP; _k++){
        wmma::load_matrix_sync(a, &As[_k][warp_k * K][warp_m * M], BLOCK_SIZE_X + SKEW);
        wmma::load_matrix_sync(b, &Bs[_k][warp_k * K][warp_n * N], BLOCK_SIZE_X + SKEW);
        wmma::mma_sync(c, a, b, c); 
        wmma::load_matrix_sync(a, &As[_k][(warp_k + KS) * K][warp_m * M], BLOCK_SIZE_X + SKEW);
        wmma::load_matrix_sync(b, &Bs[_k][(warp_k + KS) * K][warp_n * N], BLOCK_SIZE_X + SKEW);
        wmma::mma_sync(c, a, b, c); 
      }
      wmma::store_matrix_sync(&Cs[warp_k][warp_m * M][warp_n * N], 
        c, BLOCK_SIZE_X + SKEW, wmma::mem_row_major);
      // make sure all the partial sums are stored into shared memory
      __syncthreads();
      half2 to_store = *((half2*)(&Cs[0][sy][sx]));
#pragma unroll
      for (int _t = 1; _t < KS; _t++){
        to_store = __hadd2(to_store, *((half2*)(&Cs[_t][sy][sx])));
      }
      atomicAdd(((half2*)(wg_ptr + (m + sy) * c_out + (n + sx))), 
        to_store);
    }
  }
#endif
}
