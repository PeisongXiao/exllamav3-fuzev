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
        recapture(input, output, scale, bias);
    }

    void recapture(torch::Tensor input, torch::Tensor output, float scale, float bias,
                   int param_offset = 0, int param_size = sizeof(void*))
    {
        c10::cuda::CUDAGuard guard(input.device());
        GraphCapture capture(graph);
        affine<<<(input.numel() + 127) / 128, 128, 0, graph.capture_stream>>>(
            input.data_ptr<float>(), output.data_ptr<float>(), input.numel(), scale, bias);
        graph.record_param((void*) affine, GP_add_x, param_offset, param_size);
        graph.record_param((void*) affine, GP_add_y, 1);
        graph.record_param((void*) affine, GP_attn_num_pages, 2, sizeof(int));
        graph.record_param((void*) affine, GP_end, 0);
        capture.finish();
    }

    void abort()
    {
        graph.capture_abort();
    }

    void replay(torch::Tensor input, torch::Tensor output, int count = -1)
    {
        c10::cuda::CUDAGuard guard(input.device());
        if (count < 0) count = input.numel();
        graph.launch({{GP_add_x, input.data_ptr<float>()},
                      {GP_add_y, output.data_ptr<float>()},
                      {GP_attn_num_pages, (void*) (uintptr_t) count}, {GP_end, nullptr}},
                     at::cuda::getCurrentCUDAStream());
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    pybind11::class_<AffineGraph>(m, "AffineGraph")
        .def(pybind11::init<torch::Tensor, torch::Tensor, float, float>())
        .def("recapture", &AffineGraph::recapture, pybind11::arg("input"), pybind11::arg("output"),
             pybind11::arg("scale"), pybind11::arg("bias"),
             pybind11::arg("param_offset") = 0, pybind11::arg("param_size") = sizeof(void*))
        .def("abort", &AffineGraph::abort)
        .def("replay", &AffineGraph::replay, pybind11::arg("input"), pybind11::arg("output"),
             pybind11::arg("count") = -1);
}
