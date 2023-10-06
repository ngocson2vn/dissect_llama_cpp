#include "ggml-cuda.h"

void ggml_init_cublas() { printf("%s is just a stub function!\n", __func__); }

void ggml_cuda_transform_tensor(void* data, struct ggml_tensor* tensor) {
  printf("%s is just a stub function!\n", __func__);
}

void ggml_cuda_set_tensor_split(const float* tensor_split) { printf("%s is just a stub function!\n", __func__); }

void ggml_cuda_set_main_device(const int main_device) {
  (void)main_device;
  printf("%s is just a stub function!\n", __func__);
}

bool ggml_cuda_can_mul_mat(const struct ggml_tensor* src0, const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  printf("%s is just a stub function!\n", __func__);
}

void ggml_cuda_assign_buffers_no_scratch(struct ggml_tensor* tensor) {
  printf("%s is just a stub function!\n", __func__);
}

void ggml_cuda_set_scratch_size(const size_t scratch_size) { printf("%s is just a stub function!\n", __func__); }

void ggml_cuda_assign_scratch_offset(struct ggml_tensor* tensor, size_t offset) {
  printf("%s is just a stub function!\n", __func__);
}

void ggml_cuda_copy_to_device(struct ggml_tensor* tensor) { printf("%s is just a stub function!\n", __func__); }

void ggml_cuda_set_mul_mat_q(const bool mul_mat_q) { printf("%s is just a stub function!\n", __func__); }

void ggml_cuda_assign_buffers_no_alloc(struct ggml_tensor* tensor) {
  printf("%s is just a stub function!\n", __func__);
}