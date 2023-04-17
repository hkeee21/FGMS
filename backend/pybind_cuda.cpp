#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "spconv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("fgms_fusion_fwd_cuda", &ConvolutionForwardFused);
  m.def("fgms_fwd_cuda", &ConvolutionForward);
}
