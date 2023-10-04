#pragma once

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// =======================================================================
// Defines
// =======================================================================
#define GGML_MEM_ALIGN 16
#define GGML_MAX_DIMS 4
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGML_SOFT_MAX_UNROLL 4
#define GGML_VEC_DOT_UNROLL 2
#define GGML_VEC_MAD_UNROLL 32
#define GGML_MAX_OP_PARAMS 32
#define GGML_MAX_SRC 6
#define GGML_MAX_NAME 64
#define GGML_NUMA_MAX_NODES 8
#define GGML_NUMA_MAX_CPUS 512
#define GGML_MAX_CONTEXTS 64

#ifdef __cplusplus
// GGML_RESTRICT not standard in C++
#define GGML_RESTRICT
#else
#define GGML_RESTRICT restrict
#endif

// =======================================================================
// Macros
// =======================================================================
#define GGML_ASSERT(x)                                                     \
  do {                                                                     \
    if (!(x)) {                                                            \
      fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
      abort();                                                             \
    }                                                                      \
  } while (0)

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define GGML_PAD(x, n) (((x) + (n)-1) & ~((n)-1))

//
// logging
//

#if (GGML_DEBUG >= 1)
#define GGML_PRINT_DEBUG(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG(...)
#endif

#if (GGML_DEBUG >= 5)
#define GGML_PRINT_DEBUG_5(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_5(...)
#endif

#if (GGML_DEBUG >= 10)
#define GGML_PRINT_DEBUG_10(...) printf(__VA_ARGS__)
#else
#define GGML_PRINT_DEBUG_10(...)
#endif

#define GGML_PRINT(...) printf(__VA_ARGS__)

// =======================================================================
// Typedefs
// =======================================================================
#if defined(__ARM_NEON) && defined(__CUDACC__)
typedef half ggml_fp16_t;
#elif defined(__ARM_NEON)
typedef __fp16 ggml_fp16_t;
#else
typedef uint16_t ggml_fp16_t;
#endif

// floating point type used to accumulate sums
typedef double ggml_float;

typedef void (*ggml_to_float_t)(const void* GGML_RESTRICT x, float* GGML_RESTRICT y, int k);
typedef void (*ggml_from_float_t)(const float* GGML_RESTRICT x, void* GGML_RESTRICT y, int k);
typedef void (*ggml_vec_dot_t)(const int n, float* GGML_RESTRICT s, const void* GGML_RESTRICT x,
                               const void* GGML_RESTRICT y);

enum ggml_log_level { GGML_LOG_LEVEL_ERROR = 2, GGML_LOG_LEVEL_WARN = 3, GGML_LOG_LEVEL_INFO = 4 };

typedef void (*ggml_log_callback)(enum ggml_log_level level, const char* text, void* user_data);

// =======================================================================
// Data Structures
// =======================================================================
static const size_t kB = 1024;
static const size_t MB = kB * kB;
static const size_t GB = kB * kB * kB;

struct gguf_header {
  uint32_t magic;
  uint32_t version;
  uint64_t n_tensors;  // GGUFv2
  uint64_t n_kv;       // GGUFv2
};

struct gguf_str {
  uint64_t n;  // GGUFv2
  char* data;
};

enum gguf_type {
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
  GGUF_TYPE_COUNT,  // marks the end of the enum
};

union gguf_value {
  uint8_t uint8;
  int8_t int8;
  uint16_t uint16;
  int16_t int16;
  uint32_t uint32;
  int32_t int32;
  float float32;
  uint64_t uint64;
  int64_t int64;
  double float64;
  bool bool_;

  struct gguf_str str;

  struct {
    enum gguf_type type;

    uint64_t n;  // GGUFv2
    void* data;
  } arr;
};

struct gguf_kv {
  struct gguf_str key;

  enum gguf_type type;
  union gguf_value value;
};

struct gguf_context {
  struct gguf_header header;

  struct gguf_kv* kv;
  struct gguf_tensor_info* infos;

  size_t alignment;
  size_t offset;  // offset of `data` from beginning of file
  size_t size;    // size of `data` in bytes

  // uint8_t * padding;
  void* data;
};

// Designated Initializers
// https://gcc.gnu.org/onlinedocs/gcc-4.1.2/gcc/Designated-Inits.html
static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8] = sizeof(uint8_t),
    [GGUF_TYPE_INT8] = sizeof(int8_t),
    [GGUF_TYPE_UINT16] = sizeof(uint16_t),
    [GGUF_TYPE_INT16] = sizeof(int16_t),
    [GGUF_TYPE_UINT32] = sizeof(uint32_t),
    [GGUF_TYPE_INT32] = sizeof(int32_t),
    [GGUF_TYPE_FLOAT32] = sizeof(float),
    [GGUF_TYPE_BOOL] = sizeof(bool),
    [GGUF_TYPE_STRING] = sizeof(struct gguf_str),
    [GGUF_TYPE_ARRAY] = 0,  // undefined
    [GGUF_TYPE_UINT64] = sizeof(uint64_t),
    [GGUF_TYPE_INT64] = sizeof(int64_t),
    [GGUF_TYPE_FLOAT64] = sizeof(double),
};

enum ggml_type {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 (5) support has been removed
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  // k-quantizations
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_I8,
  GGML_TYPE_I16,
  GGML_TYPE_I32,
  GGML_TYPE_COUNT,
};

static const char* GGUF_TYPE_NAME[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8] = "u8",    [GGUF_TYPE_INT8] = "i8",   [GGUF_TYPE_UINT16] = "u16",  [GGUF_TYPE_INT16] = "i16",
    [GGUF_TYPE_UINT32] = "u32",  [GGUF_TYPE_INT32] = "i32", [GGUF_TYPE_FLOAT32] = "f32", [GGUF_TYPE_BOOL] = "bool",
    [GGUF_TYPE_STRING] = "str",  [GGUF_TYPE_ARRAY] = "arr", [GGUF_TYPE_UINT64] = "u64",  [GGUF_TYPE_INT64] = "i64",
    [GGUF_TYPE_FLOAT64] = "f64",
};

enum llm_kv {
  LLM_KV_GENERAL_ARCHITECTURE,
  LLM_KV_GENERAL_QUANTIZATION_VERSION,
  LLM_KV_GENERAL_ALIGNMENT,
  LLM_KV_GENERAL_NAME,
  LLM_KV_GENERAL_AUTHOR,
  LLM_KV_GENERAL_URL,
  LLM_KV_GENERAL_DESCRIPTION,
  LLM_KV_GENERAL_LICENSE,
  LLM_KV_GENERAL_SOURCE_URL,
  LLM_KV_GENERAL_SOURCE_HF_REPO,

  LLM_KV_CONTEXT_LENGTH,
  LLM_KV_EMBEDDING_LENGTH,
  LLM_KV_BLOCK_COUNT,
  LLM_KV_FEED_FORWARD_LENGTH,
  LLM_KV_USE_PARALLEL_RESIDUAL,
  LLM_KV_TENSOR_DATA_LAYOUT,

  LLM_KV_ATTENTION_HEAD_COUNT,
  LLM_KV_ATTENTION_HEAD_COUNT_KV,
  LLM_KV_ATTENTION_MAX_ALIBI_BIAS,
  LLM_KV_ATTENTION_CLAMP_KQV,
  LLM_KV_ATTENTION_LAYERNORM_EPS,
  LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,

  LLM_KV_ROPE_DIMENSION_COUNT,
  LLM_KV_ROPE_FREQ_BASE,
  LLM_KV_ROPE_SCALE_LINEAR,

  LLM_KV_TOKENIZER_MODEL,
  LLM_KV_TOKENIZER_LIST,
  LLM_KV_TOKENIZER_TOKEN_TYPE,
  LLM_KV_TOKENIZER_SCORES,
  LLM_KV_TOKENIZER_MERGES,
  LLM_KV_TOKENIZER_BOS_ID,
  LLM_KV_TOKENIZER_EOS_ID,
  LLM_KV_TOKENIZER_UNK_ID,
  LLM_KV_TOKENIZER_SEP_ID,
  LLM_KV_TOKENIZER_PAD_ID,
  LLM_KV_TOKENIZER_HF_JSON,
  LLM_KV_TOKENIZER_RWKV,
};

static std::map<llm_kv, std::string> LLM_KV_NAMES = {
    {LLM_KV_GENERAL_ARCHITECTURE, "general.architecture"},
    {LLM_KV_GENERAL_QUANTIZATION_VERSION, "general.quantization_version"},
    {LLM_KV_GENERAL_ALIGNMENT, "general.alignment"},
    {LLM_KV_GENERAL_NAME, "general.name"},
    {LLM_KV_GENERAL_AUTHOR, "general.author"},
    {LLM_KV_GENERAL_URL, "general.url"},
    {LLM_KV_GENERAL_DESCRIPTION, "general.description"},
    {LLM_KV_GENERAL_LICENSE, "general.license"},
    {LLM_KV_GENERAL_SOURCE_URL, "general.source.url"},
    {LLM_KV_GENERAL_SOURCE_HF_REPO, "general.source.huggingface.repository"},

    {LLM_KV_CONTEXT_LENGTH, "%s.context_length"},
    {LLM_KV_EMBEDDING_LENGTH, "%s.embedding_length"},
    {LLM_KV_BLOCK_COUNT, "%s.block_count"},
    {LLM_KV_FEED_FORWARD_LENGTH, "%s.feed_forward_length"},
    {LLM_KV_USE_PARALLEL_RESIDUAL, "%s.use_parallel_residual"},
    {LLM_KV_TENSOR_DATA_LAYOUT, "%s.tensor_data_layout"},

    {LLM_KV_ATTENTION_HEAD_COUNT, "%s.attention.head_count"},
    {LLM_KV_ATTENTION_HEAD_COUNT_KV, "%s.attention.head_count_kv"},
    {LLM_KV_ATTENTION_MAX_ALIBI_BIAS, "%s.attention.max_alibi_bias"},
    {LLM_KV_ATTENTION_CLAMP_KQV, "%s.attention.clamp_kqv"},
    {LLM_KV_ATTENTION_LAYERNORM_EPS, "%s.attention.layer_norm_epsilon"},
    {LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, "%s.attention.layer_norm_rms_epsilon"},

    {LLM_KV_ROPE_DIMENSION_COUNT, "%s.rope.dimension_count"},
    {LLM_KV_ROPE_FREQ_BASE, "%s.rope.freq_base"},
    {LLM_KV_ROPE_SCALE_LINEAR, "%s.rope.scale_linear"},

    {LLM_KV_TOKENIZER_MODEL, "tokenizer.ggml.model"},
    {LLM_KV_TOKENIZER_LIST, "tokenizer.ggml.tokens"},
    {LLM_KV_TOKENIZER_TOKEN_TYPE, "tokenizer.ggml.token_type"},
    {LLM_KV_TOKENIZER_SCORES, "tokenizer.ggml.scores"},
    {LLM_KV_TOKENIZER_MERGES, "tokenizer.ggml.merges"},
    {LLM_KV_TOKENIZER_BOS_ID, "tokenizer.ggml.bos_token_id"},
    {LLM_KV_TOKENIZER_EOS_ID, "tokenizer.ggml.eos_token_id"},
    {LLM_KV_TOKENIZER_UNK_ID, "tokenizer.ggml.unknown_token_id"},
    {LLM_KV_TOKENIZER_SEP_ID, "tokenizer.ggml.seperator_token_id"},
    {LLM_KV_TOKENIZER_PAD_ID, "tokenizer.ggml.padding_token_id"},
    {LLM_KV_TOKENIZER_HF_JSON, "tokenizer.huggingface.json"},
    {LLM_KV_TOKENIZER_RWKV, "tokenizer.rwkv.world"},
};

enum llm_arch {
  LLM_ARCH_LLAMA,
  LLM_ARCH_FALCON,
  LLM_ARCH_BAICHUAN,
  LLM_ARCH_GPT2,
  LLM_ARCH_GPTJ,
  LLM_ARCH_GPTNEOX,
  LLM_ARCH_MPT,
  LLM_ARCH_STARCODER,
  LLM_ARCH_UNKNOWN,
};

static std::map<std::string, llm_arch> LLM_ARCH_NAMES = {
    {"llama", LLM_ARCH_LLAMA},       {"falcon", LLM_ARCH_FALCON},       {"gpt2", LLM_ARCH_GPT2},
    {"gptj", LLM_ARCH_GPTJ},         {"gptneox", LLM_ARCH_GPTNEOX},     {"mpt", LLM_ARCH_MPT},
    {"baichuan", LLM_ARCH_BAICHUAN}, {"starcoder", LLM_ARCH_STARCODER},
};

struct gguf_tensor_info {
  struct gguf_str name;

  uint32_t n_dims;
  uint64_t ne[GGML_MAX_DIMS];

  enum ggml_type type;

  uint64_t offset;  // offset from start of `data`, must be a multiple of `ALIGNMENT`

  // for writing API
  const void* data;
  size_t size;
};

struct gguf_init_params {
  bool no_alloc;

  // if not NULL, create a ggml_context and allocate the tensor data in it
  struct ggml_context** ctx;
};

enum ggml_object_type { GGML_OBJECT_TENSOR, GGML_OBJECT_GRAPH, GGML_OBJECT_WORK_BUFFER };

// ggml object
struct ggml_object {
  size_t offs;
  size_t size;

  struct ggml_object* next;

  enum ggml_object_type type;

  char padding[4];
};

static const size_t GGML_OBJECT_SIZE = sizeof(struct ggml_object);

enum ggml_backend {
  GGML_BACKEND_CPU = 0,
  GGML_BACKEND_GPU = 10,
  GGML_BACKEND_GPU_SPLIT = 20,
};

// available tensor operations:
enum ggml_op {
  GGML_OP_NONE = 0,

  GGML_OP_DUP,
  GGML_OP_ADD,
  GGML_OP_ADD1,
  GGML_OP_ACC,
  GGML_OP_SUB,
  GGML_OP_MUL,
  GGML_OP_DIV,
  GGML_OP_SQR,
  GGML_OP_SQRT,
  GGML_OP_LOG,
  GGML_OP_SUM,
  GGML_OP_SUM_ROWS,
  GGML_OP_MEAN,
  GGML_OP_ARGMAX,
  GGML_OP_REPEAT,
  GGML_OP_REPEAT_BACK,
  GGML_OP_CONCAT,
  GGML_OP_SILU_BACK,
  GGML_OP_NORM,  // normalize
  GGML_OP_RMS_NORM,
  GGML_OP_RMS_NORM_BACK,
  GGML_OP_GROUP_NORM,

  GGML_OP_MUL_MAT,
  GGML_OP_OUT_PROD,

  GGML_OP_SCALE,
  GGML_OP_SET,
  GGML_OP_CPY,
  GGML_OP_CONT,
  GGML_OP_RESHAPE,
  GGML_OP_VIEW,
  GGML_OP_PERMUTE,
  GGML_OP_TRANSPOSE,
  GGML_OP_GET_ROWS,
  GGML_OP_GET_ROWS_BACK,
  GGML_OP_DIAG,
  GGML_OP_DIAG_MASK_INF,
  GGML_OP_DIAG_MASK_ZERO,
  GGML_OP_SOFT_MAX,
  GGML_OP_SOFT_MAX_BACK,
  GGML_OP_ROPE,
  GGML_OP_ROPE_BACK,
  GGML_OP_ALIBI,
  GGML_OP_CLAMP,
  GGML_OP_CONV_1D,
  GGML_OP_CONV_2D,
  GGML_OP_CONV_TRANSPOSE_2D,
  GGML_OP_POOL_1D,
  GGML_OP_POOL_2D,

  GGML_OP_UPSCALE,  // nearest interpolate

  GGML_OP_FLASH_ATTN,
  GGML_OP_FLASH_FF,
  GGML_OP_FLASH_ATTN_BACK,
  GGML_OP_WIN_PART,
  GGML_OP_WIN_UNPART,
  GGML_OP_GET_REL_POS,
  GGML_OP_ADD_REL_POS,

  GGML_OP_UNARY,

  GGML_OP_MAP_UNARY,
  GGML_OP_MAP_BINARY,

  GGML_OP_MAP_CUSTOM1_F32,
  GGML_OP_MAP_CUSTOM2_F32,
  GGML_OP_MAP_CUSTOM3_F32,

  GGML_OP_MAP_CUSTOM1,
  GGML_OP_MAP_CUSTOM2,
  GGML_OP_MAP_CUSTOM3,

  GGML_OP_CROSS_ENTROPY_LOSS,
  GGML_OP_CROSS_ENTROPY_LOSS_BACK,

  GGML_OP_COUNT,
};

// n-dimensional tensor
struct ggml_tensor {
  enum ggml_type type;
  enum ggml_backend backend;

  int n_dims;
  int64_t ne[GGML_MAX_DIMS];  // number of elements
  size_t nb[GGML_MAX_DIMS];   // stride in bytes:
                              // nb[0] = ggml_type_size(type)
  // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type))
  // + padding nb[i] = nb[i-1] * ne[i-1]

  // compute data
  enum ggml_op op;

  // op params - allocated as int32_t for alignment
  int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];

  bool is_param;

  struct ggml_tensor* grad;
  struct ggml_tensor* src[GGML_MAX_SRC];

  // performance
  int perf_runs;
  int64_t perf_cycles;
  int64_t perf_time_us;

  struct ggml_tensor* view_src;
  size_t view_offs;

  void* data;

  char name[GGML_MAX_NAME];

  void* extra;  // extra things e.g. for ggml-cuda.cu

  char padding[4];
};

static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);

struct ggml_init_params {
  // memory pool
  size_t mem_size;   // bytes
  void* mem_buffer;  // if NULL, memory will be allocated internally
  bool no_alloc;     // don't allocate memory for the tensor data
};

// scratch buffer
struct ggml_scratch {
  size_t offs;
  size_t size;
  void* data;
};

struct ggml_context {
  size_t mem_size;
  void* mem_buffer;
  bool mem_buffer_owned;
  bool no_alloc;
  bool no_alloc_save;  // this is used to save the no_alloc state when using
                       // scratch buffers

  int n_objects;

  struct ggml_object* objects_begin;
  struct ggml_object* objects_end;

  struct ggml_scratch scratch;
  struct ggml_scratch scratch_save;
};

struct ggml_context_container {
  bool used;

  struct ggml_context context;
};

struct ggml_numa_node {
  uint32_t cpus[GGML_NUMA_MAX_CPUS];  // hardware threads on this node
  uint32_t n_cpus;
};

struct ggml_numa_nodes {
  struct ggml_numa_node nodes[GGML_NUMA_MAX_NODES];
  uint32_t n_nodes;
  uint32_t total_cpus;  // hardware threads on system
};

struct ggml_state {
  struct ggml_context_container contexts[GGML_MAX_CONTEXTS];
  struct ggml_numa_nodes numa;
};

// WARN:
// Mis-confguration can lead to problem that's hard to reason about:
// * At best  it crash or talks nosense.
// * At worst it talks slightly difference but hard to perceive.
//
// An op has to enable INIT or FINALIZE when any of it's branch needs that pass.
// Take care about compile options (e.g., GGML_USE_xxx).
static bool GGML_OP_HAS_INIT[GGML_OP_COUNT] = {0};
static bool GGML_OP_HAS_FINALIZE[GGML_OP_COUNT] = {0};
