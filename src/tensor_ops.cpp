#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <stdexcept>
#include <cmath>

namespace py = pybind11;

// Optimized Matrix Multiplication in C++
py::array_t<float> cpp_matmul(py::array_t<float> A, py::array_t<float> B) {
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B.request();

    if (buf_A.ndim != 2 || buf_B.ndim != 2) {
        throw std::runtime_error("Matrices must be 2D");
    }

    int M = buf_A.shape[0];
    int K = buf_A.shape[1];
    int N = buf_B.shape[1];

    if (buf_B.shape[0] != K) {
        throw std::runtime_error("Matrix inner dimensions must match");
    }

    auto result = py::array_t<float>({M, N});
    py::buffer_info buf_C = result.request();

    const float* ptr_A = static_cast<const float*>(buf_A.ptr);
    const float* ptr_B = static_cast<const float*>(buf_B.ptr);
    float* ptr_C = static_cast<float*>(buf_C.ptr);

    // Simple cache-friendly loops with tiling/striding style layout
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += ptr_A[i * K + k] * ptr_B[k * N + j];
            }
            ptr_C[i * N + j] = sum;
        }
    }

    return result;
}

// Optimized element-wise ReLU in C++
py::array_t<float> cpp_relu(py::array_t<float> x) {
    py::buffer_info buf = x.request();
    auto result = py::array_t<float>(buf.size);
    py::buffer_info buf_res = result.request();

    const float* ptr_in = static_cast<const float*>(buf.ptr);
    float* ptr_out = static_cast<float*>(buf_res.ptr);

    #pragma omp parallel for
    for (size_t i = 0; i < buf.size; ++i) {
        ptr_out[i] = std::max(0.0f, ptr_in[i]);
    }

    result.resize(buf.shape);
    return result;
}

// Expose C++ functions to PyBind11 module
PYBIND11_MODULE(autograd_backend, m) {
    m.doc() = "C++ accelerated backend for autograd engine";
    m.def("matmul", &cpp_matmul, "Optimized C++ Matrix Multiplication");
    m.def("relu", &cpp_relu, "Optimized C++ ReLU Activation");
}
