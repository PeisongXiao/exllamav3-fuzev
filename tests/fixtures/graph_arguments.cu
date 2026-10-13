#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "graph.cuh"

__global__ void affine(const float* input, float* output, int count, float scale, float bias)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i] * scale + bias;
}

class AffineGraph
{
    Graph graph;

public:
    AffineGraph(torch::Tensor input, torch::Tensor output, float scale, float bias)
    {
        c10::cuda::CUDAGuard guard(input.device());
        auto stream = graph.capture_begin();
        affine<<<(input.numel() + 127) / 128, 128, 0, stream>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), input.numel(), scale, bias);
        graph.record_param((void*) affine, GP_add_x, 0);
        graph.record_param((void*) affine, GP_add_y, 1);
        graph.record_param((void*) affine, GP_end, 0);
        graph.capture_end();
    }

    void replay(torch::Tensor input, torch::Tensor output)
    {
        c10::cuda::CUDAGuard guard(input.device());
        graph.launch({{GP_add_x, input.data_ptr<float>()},
                      {GP_add_y, output.data_ptr<float>()}, {GP_end, nullptr}},
                     at::cuda::getCurrentCUDAStream());
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<AffineGraph>(m, "AffineGraph")
        .def(pybind11::init<torch::Tensor, torch::Tensor, float, float>())
        .def("replay", &AffineGraph::replay);
}
