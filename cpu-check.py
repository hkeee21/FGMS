import torch
from FGMS.backend import fgms_fwd_cuda, fgms_fusion_fwd_cuda
import os
import argparse

def kpos_quantized(knnz: torch.Tensor, kpos: torch.Tensor, k_vol: int, q: int):
    kpos_quantized = torch.zeros_like(kpos)
    for k in range(k_vol):
        kpos_quantized[k + 1] = kpos_quantized[k] \
            + torch.div(knnz[k] + q - 1, q, rounding_mode='floor') * q

    snnz_quantized = kpos_quantized[-1].cpu().int().item()

    return kpos_quantized, snnz_quantized

def cpu_compute(gpu_feats: torch.Tensor, gpu_weights: torch.Tensor, 
                out_size: int, knnz: torch.Tensor, dev_imap: torch.Tensor, 
                dev_omap: torch.Tensor, precompute: bool):
    feats = gpu_feats.cpu()
    weights = gpu_weights.cpu()
    imap = dev_imap.cpu()
    omap = dev_omap.cpu()
    k_vol = weights.size(0)
    c_in = weights.size(1)
    c_out = weights.size(2)
    mid_k = k_vol // 2 if (k_vol % 2 == 1) else 0
    output = torch.zeros((out_size, c_out), dtype=feats.dtype)
    nbaddr = 0
    for k in range(k_vol):
        nbsize = knnz[k]
        if nbsize == 0: continue
        for i in range(nbsize):
            in_index = imap[nbaddr]
            out_index = omap[nbaddr]
            for co in range(c_out):
                for ci in range(c_in):
                    output[out_index, co] += feats[in_index, ci] * weights[k, ci, co]
            nbaddr += 1
    if (precompute):
        for i in range(out_size):
            for co in range(c_out):
                for ci in range(c_in):
                    output[i, co] += feats[i, ci] * weights[mid_k, ci, co]
    return output


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

    device_capability = torch.cuda.get_device_capability()
    device_capability = device_capability[0] * 100 + device_capability[1] * 10
    arch80 = device_capability >= 800

    dir = './sample-data/tiny-samples'
    file_list = os.listdir(dir)
    for i, file in enumerate(file_list):
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
        in_channel = dev_weights.shape[1]
        out_channel = dev_weights.shape[2]
    
        separate_mid = in_nnz == out_nnz
        # create output tensor
        dev_output = torch.zeros((out_nnz, out_channel), 
            dtype=dev_feats.dtype, device=dev_feats.device)

        dev_qkpos, qsum_nnz = \
            kpos_quantized(knnz, dev_kpos, dev_weights.shape[0], 128)
        
        with torch.no_grad(): 
            if not args.fusion:
                fgms_fwd_cuda(
                    dev_feats, dev_weights, sum_nnz, dev_output, knnz.cpu(), dev_kpos, 
                    dev_imap, dev_omap, separate_mid, arch80)
            else:
                fgms_fusion_fwd_cuda(
                    dev_feats, dev_weights, qsum_nnz, dev_output, dev_kpos, dev_qkpos, 
                    dev_imap, dev_omap, separate_mid, arch80)
        
        output = dev_output.clone().cpu()
        out_sum = out_nnz * out_channel

        output_cpu = cpu_compute(
            dev_feats, dev_weights, out_nnz, knnz, dev_imap, dev_omap, separate_mid 
        )

        assert output.shape == output_cpu.shape

        allclose = torch.allclose(output, output_cpu, rtol=1e-2, atol=1e-4)

        total_error = 0
        for i in range(out_nnz):
            for j in range(out_channel):
                abs_error = abs(output[i, j] - output_cpu[i, j])
                total_error += abs_error

        print("[kernel size =(%s,%s,%s), stride=(%s,%s,%s), in channel=%d, out channel=%d] mean abs error: %.4f(%.4f/%d), all close: %s" 
            %(file[8], file[9], file[10], \
             file[13], file[14], file[15], \
             args.in_channel, args.out_channel, \
            total_error/out_sum, total_error, out_sum, allclose))
        
