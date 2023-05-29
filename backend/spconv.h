#include <math.h>
#include <stdio.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


void ConvolutionForwardFused(
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int sum_nnz, 
                        at::Tensor out_feats, 
                        const at::Tensor kpos, 
                        const at::Tensor qkpos, 
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        const bool separate_mid, 
                        const bool arch80
                        );


void ConvolutionForward(const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int sum_nnz, 
                        at::Tensor out_feats, 
                        const at::Tensor kernel_nnz, 
                        const at::Tensor kernel_pos, 
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        const bool separate_mid, 
                        const bool arch80
                        );


void ConvolutionBackward(const at::Tensor out_feats_grad, 
                        const at::Tensor in_feats, 
                        const at::Tensor kernel, 
                        const int sum_nnz, 
                        at::Tensor in_feats_grad, 
                        at::Tensor kernel_grad, 
                        const at::Tensor kpos,
                        const at::Tensor qkpos,
                        const at::Tensor in_map, 
                        const at::Tensor out_map, 
                        const bool separate_mid,
                        const bool arch80
                        );
