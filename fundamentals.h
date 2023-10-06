#pragma once

#include "common.h"

// Precomputed gelu table for f16 (128 KB)
extern ggml_fp16_t table_gelu_f16[];

// Precomputed quick gelu table for f16 (128 KB)
extern ggml_fp16_t table_gelu_quick_f16[];

// Precomputed silu table for f16 (128 KB)
extern ggml_fp16_t table_silu_f16[];

// Precomputed exp table for f16 (128 KB)
extern ggml_fp16_t table_exp_f16[];

// Precomputed f32 table for f16 (256 KB)
extern float table_f32_f16[];

// Super-block size
#ifdef GGML_QKK_64
#define QK_K 64
#define K_SCALE_SIZE 4
#else
#define QK_K 256
#define K_SCALE_SIZE 12
#endif

#ifndef static_assert
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)
#define static_assert(cond, msg) _Static_assert(cond, msg)
#else
#define static_assert(cond, msg) struct global_scope_noop_trick
#endif
#endif

#ifdef __ARM_NEON

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn
//   /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h
//   ./src/
//
#include <arm_neon.h>

#if !defined(__aarch64__)
inline static int32_t vaddvq_s16(int16x8_t v) {
  return (int32_t)vgetq_lane_s16(v, 0) + (int32_t)vgetq_lane_s16(v, 1) + (int32_t)vgetq_lane_s16(v, 2) +
         (int32_t)vgetq_lane_s16(v, 3) + (int32_t)vgetq_lane_s16(v, 4) + (int32_t)vgetq_lane_s16(v, 5) +
         (int32_t)vgetq_lane_s16(v, 6) + (int32_t)vgetq_lane_s16(v, 7);
}

inline static int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
  int16x4_t a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
  int16x4_t b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
  return vcombine_s16(a0, b0);
}

inline static int32_t vaddvq_s32(int32x4_t v) {
  return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}
#endif

#else

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#else
#ifdef __POWER9_VECTOR__
#include <altivec.h>
#undef bool
#define bool _Bool
#else
#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#else
#if !defined(__riscv)
#include <immintrin.h>
#endif
#endif
#endif
#endif
#endif

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#ifdef __cplusplus
extern "C" {
#endif

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elemenets each
// Effectively 2.5625 bits per weight
typedef struct {
  uint8_t scales[QK_K / 16];  // scales and mins, quantized with 4 bits
  uint8_t qs[QK_K / 4];       // quants
  ggml_fp16_t d;              // super-block scale for quantized scales
  ggml_fp16_t dmin;           // super-block scale for quantized mins
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2 * sizeof(ggml_fp16_t) + QK_K / 16 + QK_K / 4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elemenets each
// Effectively 3.4375 bits per weight
#ifdef GGML_QKK_64
typedef struct {
  uint8_t hmask[QK_K / 8];  // quants - high bit
  uint8_t qs[QK_K / 4];     // quants - low 2 bits
  uint8_t scales[2];
  ggml_fp16_t d;  // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + 2, "wrong q3_K block size/padding");
#else
typedef struct {
  uint8_t hmask[QK_K / 8];  // quants - high bit
  uint8_t qs[QK_K / 4];     // quants - low 2 bits
  uint8_t scales[12];       // scales, quantized with 6 bits
  ggml_fp16_t d;            // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(ggml_fp16_t) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");
#endif

// 4-bit quantization
// 16 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
#ifdef GGML_QKK_64
typedef struct {
  ggml_fp16_t d[2];      // super-block scales/mins
  uint8_t scales[2];     // 4-bit block scales/mins
  uint8_t qs[QK_K / 2];  // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2 * sizeof(ggml_fp16_t) + QK_K / 2 + 2, "wrong q4_K block size/padding");
#else
typedef struct {
  ggml_fp16_t d;                 // super-block scale for quantized scales
  ggml_fp16_t dmin;              // super-block scale for quantized mins
  uint8_t scales[K_SCALE_SIZE];  // scales and mins, quantized with 6 bits
  uint8_t qs[QK_K / 2];          // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2 * sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K / 2, "wrong q4_K block size/padding");
#endif

// 5-bit quantization
// 16 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
#ifdef GGML_QKK_64
typedef struct {
  ggml_fp16_t d;             // super-block scale
  int8_t scales[QK_K / 16];  // 8-bit block scales
  uint8_t qh[QK_K / 8];      // quants, high bit
  uint8_t qs[QK_K / 2];      // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == sizeof(ggml_fp16_t) + QK_K / 2 + QK_K / 8 + QK_K / 16,
              "wrong q5_K block size/padding");
#else
typedef struct {
  ggml_fp16_t d;                 // super-block scale for quantized scales
  ggml_fp16_t dmin;              // super-block scale for quantized mins
  uint8_t scales[K_SCALE_SIZE];  // scales and mins, quantized with 6 bits
  uint8_t qh[QK_K / 8];          // quants, high bit
  uint8_t qs[QK_K / 2];          // quants, low 4 bits
} block_q5_K;
static_assert(sizeof(block_q5_K) == 2 * sizeof(ggml_fp16_t) + K_SCALE_SIZE + QK_K / 2 + QK_K / 8,
              "wrong q5_K block size/padding");
#endif

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elemenets each
// Effectively 6.5625 bits per weight
typedef struct {
  uint8_t ql[QK_K / 2];      // quants, lower 4 bits
  uint8_t qh[QK_K / 4];      // quants, upper 2 bits
  int8_t scales[QK_K / 16];  // scales, quantized with 8 bits
  ggml_fp16_t d;             // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(ggml_fp16_t) + QK_K / 16 + 3 * QK_K / 4, "wrong q6_K block size/padding");

// This is only used for intermediate quantization and dot products
typedef struct {
  float d;                   // delta
  int8_t qs[QK_K];           // quants
  int16_t bsums[QK_K / 16];  // sum of quants in groups of 16
} block_q8_K;
static_assert(sizeof(block_q8_K) == sizeof(float) + QK_K + QK_K / 16 * sizeof(int16_t),
              "wrong q8_K block size/padding");

#define QK4_0 32
typedef struct {
  ggml_fp16_t d;          // delta
  uint8_t qs[QK4_0 / 2];  // nibbles / quants
} block_q4_0;
static_assert(sizeof(block_q4_0) == sizeof(ggml_fp16_t) + QK4_0 / 2, "wrong q4_0 block size/padding");

#define QK4_1 32
typedef struct {
  ggml_fp16_t d;          // delta
  ggml_fp16_t m;          // min
  uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
static_assert(sizeof(block_q4_1) == 2 * sizeof(ggml_fp16_t) + QK4_1 / 2, "wrong q4_1 block size/padding");

#define QK5_0 32
typedef struct {
  ggml_fp16_t d;          // delta
  uint8_t qh[4];          // 5-th bit of quants
  uint8_t qs[QK5_0 / 2];  // nibbles / quants
} block_q5_0;
static_assert(sizeof(block_q5_0) == sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_0 / 2,
              "wrong q5_0 block size/padding");

#define QK5_1 32
typedef struct {
  ggml_fp16_t d;          // delta
  ggml_fp16_t m;          // min
  uint8_t qh[4];          // 5-th bit of quants
  uint8_t qs[QK5_1 / 2];  // nibbles / quants
} block_q5_1;
static_assert(sizeof(block_q5_1) == 2 * sizeof(ggml_fp16_t) + sizeof(uint32_t) + QK5_1 / 2,
              "wrong q5_1 block size/padding");

#define QK8_0 32
typedef struct {
  ggml_fp16_t d;     // delta
  int8_t qs[QK8_0];  // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

#define QK8_1 32
typedef struct {
  float d;           // delta
  float s;           // d * sum(qs[i])
  int8_t qs[QK8_1];  // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2 * sizeof(float) + QK8_1, "wrong q8_1 block size/padding");

// Quantization
void quantize_row_q2_K_reference(const float* GGML_RESTRICT x, block_q2_K* GGML_RESTRICT y, int k);
void quantize_row_q3_K_reference(const float* GGML_RESTRICT x, block_q3_K* GGML_RESTRICT y, int k);
void quantize_row_q4_K_reference(const float* GGML_RESTRICT x, block_q4_K* GGML_RESTRICT y, int k);
void quantize_row_q5_K_reference(const float* GGML_RESTRICT x, block_q5_K* GGML_RESTRICT y, int k);
void quantize_row_q6_K_reference(const float* GGML_RESTRICT x, block_q6_K* GGML_RESTRICT y, int k);
void quantize_row_q8_K_reference(const float* GGML_RESTRICT x, block_q8_K* GGML_RESTRICT y, int k);

void quantize_row_q2_K(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q3_K(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q4_K(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q5_K(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q6_K(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q8_K(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);

void quantize_row_q4_0(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q4_1(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q5_0(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q5_1(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
void quantize_row_q8_0(const float* GGML_RESTRICT x, void* GGML_RESTRICT vy, int k);
void quantize_row_q8_1(const float* GGML_RESTRICT x, void* GGML_RESTRICT vy, int k);

void quantize_row_q4_0_reference(const float* GGML_RESTRICT x, block_q4_0* GGML_RESTRICT y, int k);
void quantize_row_q4_1_reference(const float* GGML_RESTRICT x, block_q4_1* GGML_RESTRICT y, int k);
void quantize_row_q5_0_reference(const float* GGML_RESTRICT x, block_q5_0* GGML_RESTRICT y, int k);
void quantize_row_q5_1_reference(const float* GGML_RESTRICT x, block_q5_1* GGML_RESTRICT y, int k);
void quantize_row_q8_0_reference(const float* GGML_RESTRICT x, block_q8_0* GGML_RESTRICT y, int k);
void quantize_row_q8_1_reference(const float* GGML_RESTRICT x, block_q8_1* GGML_RESTRICT y, int k);

// Dequantization
void dequantize_row_q2_K(const block_q2_K* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q3_K(const block_q3_K* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q4_K(const block_q4_K* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q5_K(const block_q5_K* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q6_K(const block_q6_K* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q8_K(const block_q8_K* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);

void dequantize_row_q4_0(const block_q4_0* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q4_1(const block_q4_1* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q5_0(const block_q5_0* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q5_1(const block_q5_1* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
void dequantize_row_q8_0(const void* GGML_RESTRICT vx, float* GGML_RESTRICT y, int k);

// Dot product
void ggml_vec_dot_q2_K_q8_K(int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy);
void ggml_vec_dot_q3_K_q8_K(int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy);
void ggml_vec_dot_q4_K_q8_K(int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy);
void ggml_vec_dot_q5_K_q8_K(int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy);
void ggml_vec_dot_q6_K_q8_K(int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx, const void* GGML_RESTRICT vy);

// Quantization with histogram collection
size_t ggml_quantize_q2_K(const float* src, void* dst, int n, int k, int64_t* hist);
size_t ggml_quantize_q3_K(const float* src, void* dst, int n, int k, int64_t* hist);
size_t ggml_quantize_q4_K(const float* src, void* dst, int n, int k, int64_t* hist);
size_t ggml_quantize_q5_K(const float* src, void* dst, int n, int k, int64_t* hist);
size_t ggml_quantize_q6_K(const float* src, void* dst, int n, int k, int64_t* hist);

void ggml_vec_dot_f32(const int n, float* GGML_RESTRICT s, const float* GGML_RESTRICT x, const float* GGML_RESTRICT y);
void ggml_vec_dot_f16(const int n, float* GGML_RESTRICT s, ggml_fp16_t* GGML_RESTRICT x, ggml_fp16_t* GGML_RESTRICT y);
void ggml_vec_dot_q4_0_q8_0(const int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx,
                            const void* GGML_RESTRICT vy);
void ggml_vec_dot_q4_1_q8_1(const int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx,
                            const void* GGML_RESTRICT vy);
void ggml_vec_dot_q5_0_q8_0(const int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx,
                            const void* GGML_RESTRICT vy);
void ggml_vec_dot_q5_1_q8_1(const int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx,
                            const void* GGML_RESTRICT vy);
void ggml_vec_dot_q8_0_q8_0(const int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT vx,
                            const void* GGML_RESTRICT vy);

// Fundamental ops
float ggml_fp16_to_fp32(ggml_fp16_t x);
ggml_fp16_t ggml_fp32_to_fp16(float x);
void ggml_fp16_to_fp32_row(const ggml_fp16_t* x, float* y, int n);
void ggml_fp32_to_fp16_row(const float* x, ggml_fp16_t* y, int n);
float ggml_compute_fp16_to_fp32(ggml_fp16_t h);
ggml_fp16_t ggml_compute_fp32_to_fp16(float f);
float ggml_lookup_fp16_to_fp32(ggml_fp16_t f);

float ggml_gelu_f32(float x);
float ggml_gelu_quick_f32(float x);
float ggml_silu_f32(float x);
float ggml_silu_backward_f32(float x, float dy);

void ggml_vec_set_i8(const int n, int8_t* x, const int8_t v);
void ggml_vec_set_i16(const int n, int16_t* x, const int16_t v);
void ggml_vec_set_i32(const int n, int32_t* x, const int32_t v);
void ggml_vec_set_f16(const int n, ggml_fp16_t* x, const int32_t v);
void ggml_vec_set_f32(const int n, float* x, const float v);
void ggml_vec_add_f32(const int n, float* z, const float* x, const float* y);
void ggml_vec_acc_f32(const int n, float* y, const float* x);
void ggml_vec_add1_f32(const int n, float* z, const float* x, const float v);
void ggml_vec_acc1_f32(const int n, float* y, const float v);
void ggml_vec_sub_f32(const int n, float* z, const float* x, const float* y);
void ggml_vec_mul_f32(const int n, float* z, const float* x, const float* y);
void ggml_vec_div_f32(const int n, float* z, const float* x, const float* y);
void ggml_vec_sqr_f32(const int n, float* y, const float* x);
void ggml_vec_sqrt_f32(const int n, float* y, const float* x);
void ggml_vec_log_f32(const int n, float* y, const float* x);
void ggml_vec_sum_f32_ggf(const int n, ggml_float* s, const float* x);
void ggml_vec_sum_f16_ggf(const int n, float* s, const ggml_fp16_t* x);
void ggml_vec_sum_f32(const int n, float* s, const float* x);
void ggml_vec_argmax_f32(const int n, int* s, const float* x);
void ggml_vec_cpy_f32(const int n, float* y, const float* x);
void ggml_vec_abs_f32(const int n, float* y, const float* x);
void ggml_vec_neg_f32(const int n, float* y, const float* x);
void ggml_vec_sgn_f32(const int n, float* y, const float* x);
void ggml_vec_step_f32(const int n, float* y, const float* x);
void ggml_vec_tanh_f32(const int n, float* y, const float* x);
void ggml_vec_elu_f32(const int n, float* y, const float* x);
void ggml_vec_relu_f32(const int n, float* y, const float* x);
void ggml_vec_gelu_f32(const int n, float* y, const float* x);
void ggml_vec_silu_f32(const int n, float* y, const float* x);
void ggml_vec_silu_backward_f32(const int n, float* dx, const float* x, const float* dy);
void ggml_vec_scale_f32(const int n, float* y, const float v);
void ggml_vec_mad_f32_unroll(const int n, const int xs, const int vs, float* GGML_RESTRICT y,
                             const float* GGML_RESTRICT xv, const float* GGML_RESTRICT vv);
void ggml_vec_mad_f32(const int n, float* GGML_RESTRICT y, const float* GGML_RESTRICT x, const float v);
void ggml_vec_gelu_quick_f32(const int n, float* y, const float* x);
void ggml_vec_max_f32(const int n, float* s, const float* x);
void ggml_vec_dot_f16_unroll(const int n, const int xs, float* GGML_RESTRICT s, void* GGML_RESTRICT xv,
                             ggml_fp16_t* GGML_RESTRICT y);
void ggml_vec_gelu_f16(const int n, ggml_fp16_t* y, const ggml_fp16_t* x);

#ifdef __cplusplus
}
#endif

// 16-bit float
// on Arm, we use __fp16
// on x86, we use uint16_t
#if defined(__ARM_NEON)

// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn
//   /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h
//   ./src/
//
#include <arm_neon.h>

#define GGML_COMPUTE_FP16_TO_FP32(x) ((float)(x))
#define GGML_COMPUTE_FP32_TO_FP16(x) (x)

#define GGML_FP16_TO_FP32(x) ((float)(x))
#define GGML_FP32_TO_FP16(x) (x)

#endif

// On ARM NEON, it's quicker to directly convert x -> x instead of calling into
// ggml_lookup_fp16_to_fp32, so we define GGML_FP16_TO_FP32 and
// GGML_FP32_TO_FP16 elsewhere for NEON. This is also true for POWER9.
#if !defined(GGML_FP16_TO_FP32) || !defined(GGML_FP32_TO_FP16)

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) ggml_lookup_fp16_to_fp32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

#endif
