import torch
from FGMS.backend import fgms_bwd_cuda
import os
import argparse

def kpos_quantized(knnz: torch.Tensor, kpos: torch.Tensor, k_vol: int, q: int):
    kpos_quantized = torch.zeros_like(kpos)
    for k in range(k_vol):
        kpos_quantized[k + 1] = kpos_quantized[k] \
            + torch.div(knnz[k] + q - 1, q, rounding_mode='floor') * q

    snnz_quantized = kpos_quantized[-1].cpu().int().item()

    return kpos_quantized, snnz_quantized

def cpu_compute(gpu_out_grad: torch.Tensor, gpu_weights: torch.Tensor, 
                gpu_in_feats: torch.Tensor, 
                in_size: int, knnz: torch.Tensor, dev_imap: torch.Tensor, 
                dev_omap: torch.Tensor):
    out_grad = gpu_out_grad.cpu()
    weights = gpu_weights.cpu()
    in_feats = gpu_in_feats.cpu()
    imap = dev_imap.cpu()
    omap = dev_omap.cpu()
    k_vol = weights.size(0)
    c_in = weights.size(1)
    c_out = weights.size(2)
    in_grad = torch.zeros((in_size, c_in), dtype=out_grad.dtype)
    weight_grad = torch.zeros((k_vol, c_in, c_out), dtype=weights.dtype)   
    nbaddr = 0
    for k in range(k_vol):
        nbsize = knnz[k]
        if nbsize == 0: continue
        for i in range(nbsize):
            in_index = imap[nbaddr]
            out_index = omap[nbaddr]
            for ci in range(c_in):
                for co in range(c_out):
                    in_grad[in_index, ci] += out_grad[out_index, co] * weights[k, ci, co]
                    weight_grad[k, ci, co] += in_feats[in_index, ci] * out_grad[out_index, co]
                    
            nbaddr += 1
    
    return in_grad, weight_grad


def channel_transform(feats: torch.Tensor, weights: torch.Tensor, 
                      ci_tgt: int, co_tgt: int):
    ci_orig = weights.size(1)
    co_orig = weights.size(2)

    while(ci_orig < ci_tgt):
        feats = torch.cat([feats, feats], dim=1)
        weights = torch.cat([weights, weights], dim=1)
        ci_orig *= 2
    while(co_orig < co_tgt):
        weights = torch.cat([weights, weights], dim=2)
        co_orig *= 2
    feats = feats[:, 0:ci_tgt]
    weights = weights[:, 0:ci_tgt, 0:co_tgt]

    feats = feats.contiguous()
    weights = weights.contiguous()

    return feats, weights


if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-channel', type=int, default=16)
    parser.add_argument('--out-channel', type=int, default=16)
    parser.add_argument('--fusion', default=False, action='store_true')
    args = parser.parse_args()
    dir = './sample-data/tiny-samples'
    file_list = os.listdir(dir)
    print(file_list)
    for i, file in enumerate(file_list):
        if 'FP16' in file: continue
        # loading data info from file ...
        conv_data = torch.load(os.path.join(dir, file))
        dev_feats = conv_data['in_feats']
        dev_weights = conv_data['kernel']
        dev_feats, dev_weights = channel_transform(
            dev_feats, dev_weights, args.in_channel, args.out_channel
        )
        sum_nnz = conv_data['sum_nnz']
        out_nnz = conv_data['out_nnz']
        knnz = conv_data['knnz']
        dev_kpos = conv_data['kpos']
        dev_imap = conv_data['imap'].reshape(-1)
        dev_omap = conv_data['omap'].reshape(-1)

        in_nnz = dev_feats.shape[0]
        kernel_vol = dev_weights.shape[0]
        in_channel = dev_weights.shape[1]
        out_channel = dev_weights.shape[2]
    
        separate_mid = in_nnz == out_nnz

        # create random out_grad
        dev_out_grad = torch.rand((out_nnz, out_channel), dtype=dev_feats.dtype, 
                            device=dev_feats.device)
        dev_weight_grad = torch.zeros_like(dev_weights)

        # create output tensor
        dev_in_grad = torch.zeros((in_nnz, in_channel), 
            dtype=dev_feats.dtype, device=dev_feats.device)

        dev_qkpos, qsum_nnz = \
            kpos_quantized(knnz, dev_kpos, dev_weights.shape[0], 128)
        
        with torch.no_grad(): 
            fgms_bwd_cuda(dev_out_grad, dev_feats, dev_weights, qsum_nnz, 
                          dev_in_grad, dev_weight_grad, dev_kpos, dev_qkpos, 
                          dev_imap, dev_omap, False, False)
        
        in_grad = dev_in_grad.clone().cpu()
        weight_grad = dev_weight_grad.clone().cpu()

        in_grad_cpu, weight_grad_cpu = cpu_compute(
            dev_out_grad, dev_weights, dev_feats, in_nnz, knnz, dev_imap, dev_omap
        )

        assert in_grad.shape == in_grad_cpu.shape

        allclose = torch.allclose(in_grad, in_grad_cpu, rtol=1e-2, atol=1e-4)

        total_error = 0
        for i in range(in_nnz):
            for j in range(in_channel):
                abs_error = abs(in_grad[i, j] - in_grad_cpu[i, j])
                total_error += abs_error

        print("[kernel size =(%s,%s,%s), stride=(%s,%s,%s), in channel=%d, out channel=%d] mean abs error: %.4f(%.4f/%d), all close: %s" 
            %(file[8], file[9], file[10], \
             file[13], file[14], file[15], \
             args.in_channel, args.out_channel, \
            total_error/(in_nnz * in_channel), total_error, (in_nnz * in_channel), allclose))
        
        assert weight_grad.shape == weight_grad_cpu.shape

        allclose = torch.allclose(weight_grad, weight_grad_cpu, rtol=1e-2, atol=1e-4)

        total_error = 0
        for k in range(kernel_vol):
            for i in range(in_channel):
                for j in range(out_channel):
                    abs_error = abs(weight_grad[k, i, j] - weight_grad_cpu[k, i, j])
                    total_error += abs_error

        print("[kernel size =(%s,%s,%s), stride=(%s,%s,%s), in channel=%d, out channel=%d] mean abs error: %.4f(%.4f/%d), all close: %s" 
            %(file[8], file[9], file[10], \
             file[13], file[14], file[15], \
             args.in_channel, args.out_channel, \
                total_error/(kernel_vol * in_channel * in_channel), \
                total_error, (kernel_vol * in_channel * in_channel), allclose))
        
