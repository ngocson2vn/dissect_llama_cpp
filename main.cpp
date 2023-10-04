#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "fundamentals.h"
#include "llama.h"

// =======================================================================
// Global state
// =======================================================================
static struct ggml_state g_state;
// static atomic_int g_state_barrier = 0;

static struct ggml_context* ctx_meta = NULL;

// =======================================================================
// Utility functions
// =======================================================================
static void* ggml_aligned_malloc(size_t size) {
  void* aligned_memory = NULL;
  int result = posix_memalign(&aligned_memory, GGML_MEM_ALIGN, size);
  if (result != 0) {
    fprintf(stderr, "Could not allocate memory chunk with size = %d and alignment = %d\n", size, GGML_MEM_ALIGN);
  }

  return aligned_memory;
}

void gguf_free(struct gguf_context* gguf_ctx) {
  if (gguf_ctx == NULL) {
    return;
  }

  if (gguf_ctx->kv) {
    // free string memory - not great..
    for (uint32_t i = 0; i < gguf_ctx->header.n_kv; ++i) {
      struct gguf_kv* kv = &gguf_ctx->kv[i];

      if (kv->key.data) {
        free(kv->key.data);
      }

      if (kv->type == GGUF_TYPE_STRING) {
        if (kv->value.str.data) {
          free(kv->value.str.data);
        }
      }

      if (kv->type == GGUF_TYPE_ARRAY) {
        if (kv->value.arr.data) {
          if (kv->value.arr.type == GGUF_TYPE_STRING) {
            for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
              struct gguf_str* str = &((struct gguf_str*)kv->value.arr.data)[j];
              if (str->data) {
                free(str->data);
              }
            }
          }
          free(kv->value.arr.data);
        }
      }
    }

    free(gguf_ctx->kv);
  }

  if (gguf_ctx->infos) {
    for (uint32_t i = 0; i < gguf_ctx->header.n_tensors; ++i) {
      struct gguf_tensor_info* info = &gguf_ctx->infos[i];

      if (info->name.data) {
        free(info->name.data);
      }
    }

    free(gguf_ctx->infos);
  }

  free(gguf_ctx);
}

static void ggml_setup_op_has_task_pass(void) {
  {  // INIT
    bool* p = GGML_OP_HAS_INIT;

    p[GGML_OP_ACC] = true;
    p[GGML_OP_MUL_MAT] = true;
    p[GGML_OP_OUT_PROD] = true;
    p[GGML_OP_SET] = true;
    p[GGML_OP_GET_ROWS_BACK] = true;
    p[GGML_OP_DIAG_MASK_INF] = true;
    p[GGML_OP_DIAG_MASK_ZERO] = true;
    p[GGML_OP_CONV_1D] = true;
    p[GGML_OP_CONV_2D] = true;
    p[GGML_OP_CONV_TRANSPOSE_2D] = true;
    p[GGML_OP_FLASH_ATTN_BACK] = true;
    p[GGML_OP_CROSS_ENTROPY_LOSS] = true;
    p[GGML_OP_ADD_REL_POS] = true;
  }

  {  // FINALIZE
    bool* p = GGML_OP_HAS_FINALIZE;

    p[GGML_OP_CROSS_ENTROPY_LOSS] = true;
  }
}

struct ggml_context* ggml_init(struct ggml_init_params params) {
  // make this function thread safe
  // ggml_critical_section_start();

  static bool is_first_call = true;

  if (is_first_call) {
    // initialize time system (required on Windows)
    // ggml_time_init();

    // initialize GELU, Quick GELU, SILU and EXP F32 tables
    {
      // const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

      ggml_fp16_t ii;
      for (int i = 0; i < (1 << 16); ++i) {
        uint16_t ui = i;
        memcpy(&ii, &ui, sizeof(ii));
        const float f = table_f32_f16[i] = GGML_COMPUTE_FP16_TO_FP32(ii);
        table_gelu_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_f32(f));
        table_gelu_quick_f16[i] = GGML_FP32_TO_FP16(ggml_gelu_quick_f32(f));
        table_silu_f16[i] = GGML_FP32_TO_FP16(ggml_silu_f32(f));
        table_exp_f16[i] = GGML_FP32_TO_FP16(expf(f));
      }

      // const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

      // GGML_PRINT_DEBUG("%s: GELU, Quick GELU, SILU and EXP tables initialized
      // in %f ms\n", __func__, (t_end - t_start)/1000.0f);
    }

    // initialize g_state
    {
      // const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

      g_state = (struct ggml_state){
          /*.contexts =*/{{0}},
          /*.numa =*/
          {
              .n_nodes = 0,
              .total_cpus = 0,
          },
      };

      for (int i = 0; i < GGML_MAX_CONTEXTS; ++i) {
        g_state.contexts[i].used = false;
      }

      // const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

      // GGML_PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end
      // - t_start)/1000.0f);
    }

    // #if defined(GGML_USE_CUBLAS)
    //         ggml_init_cublas();
    // #elif defined(GGML_USE_CLBLAST)
    //         ggml_cl_init();
    // #endif

    ggml_setup_op_has_task_pass();

    is_first_call = false;
  }

  // find non-used context in g_state
  struct ggml_context* ctx = NULL;

  for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
    if (!g_state.contexts[i].used) {
      g_state.contexts[i].used = true;
      ctx = &g_state.contexts[i].context;

      // GGML_PRINT_DEBUG("%s: found unused context %d\n", __func__, i);
      break;
    }
  }

  if (ctx == NULL) {
    // GGML_PRINT_DEBUG("%s: no unused context found\n", __func__);

    // ggml_critical_section_end();

    return NULL;
  }

  // allow to call ggml_init with 0 size
  if (params.mem_size == 0) {
    params.mem_size = GGML_MEM_ALIGN;
  }

  const size_t mem_size = params.mem_buffer ? params.mem_size : GGML_PAD(params.mem_size, GGML_MEM_ALIGN);

  *ctx = (struct ggml_context){
      /*.mem_size           =*/mem_size,
      /*.mem_buffer         =*/params.mem_buffer ? params.mem_buffer : ggml_aligned_malloc(mem_size),
      /*.mem_buffer_owned   =*/params.mem_buffer ? false : true,
      /*.no_alloc           =*/params.no_alloc,
      /*.no_alloc_save      =*/params.no_alloc,
      /*.n_objects          =*/0,
      /*.objects_begin      =*/NULL,
      /*.objects_end        =*/NULL,
      /*.scratch            =*/
      {
          0,
          0,
          NULL,
      },
      /*.scratch_save       =*/
      {
          0,
          0,
          NULL,
      },
  };

  GGML_ASSERT(ctx->mem_buffer != NULL);

  // ggml_assert_aligned(ctx->mem_buffer);
  GGML_ASSERT(((uintptr_t)(ctx->mem_buffer)) % GGML_MEM_ALIGN == 0);

  // GGML_PRINT_DEBUG("%s: context initialized\n", __func__);

  // ggml_critical_section_end();

  return ctx;
}

size_t ggml_nbytes(const struct ggml_tensor* tensor) {
  size_t nbytes;
  size_t blck_size = type_traits[tensor->type].blck_size;
  if (blck_size == 1) {
    nbytes = type_traits[tensor->type].type_size;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
      nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  } else {
    nbytes = tensor->ne[0] * tensor->nb[0] / blck_size;
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
      nbytes += (tensor->ne[i] - 1) * tensor->nb[i];
    }
  }

  return nbytes;
}

static struct ggml_object* ggml_new_object(struct ggml_context* ctx, enum ggml_object_type type, size_t size) {
  // always insert objects at the end of the context's memory pool
  struct ggml_object* obj_cur = ctx->objects_end;

  const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
  const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
  const size_t cur_end = cur_offs + cur_size;

  // align to GGML_MEM_ALIGN
  size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);

  char* const mem_buffer = (char*)ctx->mem_buffer;
  struct ggml_object* const obj_new = (struct ggml_object*)(mem_buffer + cur_end);

  if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) {
    // GGML_PRINT("%s: not enough space in the context's memory pool (needed
    // %zu, available %zu)\n",
    //         __func__, cur_end + size_needed, ctx->mem_size);
    assert(false);
    return NULL;
  }

  *obj_new = (struct ggml_object){
      .offs = cur_end + GGML_OBJECT_SIZE,
      .size = size_needed,
      .next = NULL,
      .type = type,
  };

  // ggml_assert_aligned(mem_buffer + obj_new->offs);
  GGML_ASSERT(((uintptr_t)(mem_buffer + obj_new->offs)) % GGML_MEM_ALIGN == 0);

  if (obj_cur != NULL) {
    obj_cur->next = obj_new;
  } else {
    // this is the first object in this context
    ctx->objects_begin = obj_new;
  }

  ctx->objects_end = obj_new;

  // printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end,
  // obj_new->size);

  return obj_new;
}

static struct ggml_tensor* ggml_new_tensor_impl(struct ggml_context* ctx, enum ggml_type type, int n_dims,
                                                const int64_t* ne, struct ggml_tensor* view_src, size_t view_offs) {
  assert(n_dims >= 1 && n_dims <= GGML_MAX_DIMS);

  // find the base tensor and absolute offset
  if (view_src != NULL && view_src->view_src != NULL) {
    view_offs += view_src->view_offs;
    view_src = view_src->view_src;
  }

  size_t data_size = type_traits[type].type_size * (ne[0] / type_traits[type].blck_size);
  for (int i = 1; i < n_dims; i++) {
    data_size *= ne[i];
  }

  GGML_ASSERT(view_src == NULL || data_size + view_offs <= ggml_nbytes(view_src));

  void* data = view_src != NULL ? view_src->data : NULL;
  if (data != NULL) {
    data = (char*)data + view_offs;
  }

  size_t obj_alloc_size = 0;

  if (view_src == NULL && !ctx->no_alloc) {
    if (ctx->scratch.data != NULL) {
      // allocate tensor data in the scratch buffer
      if (ctx->scratch.offs + data_size > ctx->scratch.size) {
        // GGML_PRINT("%s: not enough space in the scratch memory pool (needed
        // %zu, available %zu)\n",
        //         __func__, ctx->scratch.offs + data_size, ctx->scratch.size);
        assert(false);
        return NULL;
      }

      data = (char* const)ctx->scratch.data + ctx->scratch.offs;

      ctx->scratch.offs += data_size;
    } else {
      // allocate tensor data in the context's memory pool
      obj_alloc_size = data_size;
    }
  }

  struct ggml_object* const obj_new =
      ggml_new_object(ctx, (ggml_object_type)GGML_OBJECT_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);

  // TODO: for recoverable errors, we would need to free the data allocated from
  // the scratch buffer here

  struct ggml_tensor* const result = (struct ggml_tensor*)((char*)ctx->mem_buffer + obj_new->offs);

  *result = (struct ggml_tensor){
      /*.type         =*/type,
      /*.backend      =*/(ggml_backend)GGML_BACKEND_CPU,
      /*.n_dims       =*/n_dims,
      /*.ne           =*/{1, 1, 1, 1},
      /*.nb           =*/{0, 0, 0, 0},
      /*.op           =*/(ggml_op)GGML_OP_NONE,
      /*.op_params    =*/{0},
      /*.is_param     =*/false,
      /*.grad         =*/NULL,
      /*.src          =*/{NULL},
      /*.perf_runs    =*/0,
      /*.perf_cycles  =*/0,
      /*.perf_time_us =*/0,
      /*.view_src     =*/view_src,
      /*.view_offs    =*/view_offs,
      /*.data         =*/obj_alloc_size > 0 ? (void*)(result + 1) : data,
      /*.name         =*/{0},
      /*.extra        =*/NULL,
      /*.padding      =*/{0},
  };

  // TODO: this should not be needed as long as we don't rely on aligned SIMD loads
  // ggml_assert_aligned(result->data);

  for (int i = 0; i < n_dims; i++) {
    result->ne[i] = ne[i];
  }

  result->nb[0] = type_traits[type].type_size;
  result->nb[1] = result->nb[0] * (result->ne[0] / type_traits[type].blck_size);
  for (int i = 2; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = result->nb[i - 1] * result->ne[i - 1];
  }

  ctx->n_objects++;

  return result;
}

struct ggml_tensor* ggml_new_tensor(struct ggml_context* ctx, enum ggml_type type, int n_dims, const int64_t* ne) {
  return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL, 0);
}

struct ggml_tensor* ggml_set_name(struct ggml_tensor* tensor, const char* name) {
  strncpy(tensor->name, name, sizeof(tensor->name));
  tensor->name[sizeof(tensor->name) - 1] = '\0';
  return tensor;
}

void ggml_free(struct ggml_context* ctx) {
  // make this function thread safe
  // ggml_critical_section_start();

  bool found = false;

  for (int i = 0; i < GGML_MAX_CONTEXTS; i++) {
    if (&g_state.contexts[i].context == ctx) {
      g_state.contexts[i].used = false;

      GGML_PRINT_DEBUG("%s: context %d has been freed. memory used = %zu\n", __func__, i, ggml_used_mem(ctx));

      if (ctx->mem_buffer_owned) {
        free(ctx->mem_buffer);
      }

      found = true;
      break;
    }
  }

  if (!found) {
    GGML_PRINT_DEBUG("%s: context not found\n", __func__);
  }

  // ggml_critical_section_end();
}

struct ggml_tensor* ggml_get_tensor(struct ggml_context* ctx, const char* name) {
  struct ggml_object* obj = ctx->objects_begin;

  char* const mem_buffer = (char*)ctx->mem_buffer;

  while (obj != NULL) {
    if (obj->type == GGML_OBJECT_TENSOR) {
      struct ggml_tensor* cur = (struct ggml_tensor*)(mem_buffer + obj->offs);
      if (strcmp(cur->name, name) == 0) {
        return cur;
      }
    }

    obj = obj->next;
  }

  return NULL;
}

struct ggml_tensor* create_tensor_for(struct ggml_context* ctx, struct ggml_tensor* meta, ggml_backend backend) {
  if (backend != GGML_BACKEND_CPU) {
    ctx->no_alloc = true;
  }

  struct ggml_tensor* tensor = ggml_new_tensor(ctx, meta->type, meta->n_dims, meta->ne);
  tensor->backend = backend;
  ggml_set_name(tensor, meta->name);

  if (backend != GGML_BACKEND_CPU) {
    ctx->no_alloc = true;
  }

  return tensor;
}

int gguf_find_key(const struct gguf_context* gguf_ctx, const char* key) {
  // return -1 if key not found
  int keyfound = -1;

  const uint64_t n_kv = gguf_ctx->header.n_kv;

  for (uint64_t i = 0; i < n_kv; ++i) {
    if (strcmp(key, gguf_ctx->kv[i].key.data) == 0) {
      keyfound = i;
      break;
    }
  }

  return keyfound;
}

bool ggml_is_numa(void) { return g_state.numa.n_nodes > 1; }

int gguf_find_tensor(const struct gguf_context* ctx, const char* name) {
  // return -1 if tensor not found
  int tensorfound = -1;

  for (int i = 0; i < ctx->header.n_tensors; ++i) {
    if (strcmp(name, ctx->infos[i].name.data) == 0) {
      tensorfound = i;
      break;
    }
  }

  return tensorfound;
}

static bool ggml_use_cuda() { return std::getenv("GGML_USE_CUDA") ? true : false; }

static size_t utf8_len(char src) {
  const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
  uint8_t highbits = static_cast<uint8_t>(src) >> 4;
  return lookup[highbits];
}

static void replace_all(std::string& s, const std::string& search, const std::string& replace) {
  std::string result;
  for (size_t pos = 0;; pos += search.length()) {
    auto new_pos = s.find(search, pos);
    if (new_pos == std::string::npos) {
      result += s.substr(pos, s.size() - pos);
      break;
    }
    result += s.substr(pos, new_pos - pos) + replace;
    pos = new_pos;
  }
  s = std::move(result);
}

static void zeros(std::ofstream& file, size_t n) {
  char zero = 0;
  for (size_t i = 0; i < n; ++i) {
    file.write(&zero, 1);
  }
}

std::string build_llm_key(const std::string& original_key, const std::string& arch_name) {
  if (original_key.find("%s") >= 0) {
    char* key = (char*)calloc((original_key.length() - 2) + arch_name.length() + 1, 1);
    sprintf(key, original_key.c_str(), arch_name.c_str());
    return key;
  }

  return original_key;
}

#ifdef __GNUC__
#ifdef __MINGW32__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#else
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#endif
#else
#define LLAMA_ATTRIBUTE_FORMAT(...)
#endif

LLAMA_ATTRIBUTE_FORMAT(1, 2)
static std::string format(const char* fmt, ...) {
  va_list ap;
  va_list ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);
  int size = vsnprintf(NULL, 0, fmt, ap);
  GGML_ASSERT(size >= 0 && size < INT_MAX);  // NOLINT
  std::vector<char> buf(size + 1);
  int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
  GGML_ASSERT(size2 == size);
  va_end(ap2);
  va_end(ap);
  return std::string(buf.data(), size);
}

// =======================================================================
// LLaMa Model
// =======================================================================
#ifdef GGML_USE_CUBLAS
#define llama_host_malloc(n) ggml_cuda_host_malloc(n)
#define llama_host_free(data) ggml_cuda_host_free(data)
#elif GGML_USE_METAL
#define llama_host_malloc(n) ggml_metal_host_malloc(n)
#define llama_host_free(data) ggml_metal_host_free(data)
#elif GGML_USE_CPU_HBM
#define llama_host_malloc(n) hbw_malloc(n)
#define llama_host_free(data) \
  if (data != NULL) hbw_free(data)
#else
#define llama_host_malloc(n) malloc(n)
#define llama_host_free(data) free(data)
#endif

// available llama models
enum e_model {
  MODEL_UNKNOWN,
  MODEL_1B,
  MODEL_3B,
  MODEL_7B,
  MODEL_13B,
  MODEL_15B,
  MODEL_30B,
  MODEL_34B,
  MODEL_40B,
  MODEL_65B,
  MODEL_70B,
};

struct llama_hparams {
  bool vocab_only;
  uint32_t n_vocab;
  uint32_t n_ctx_train;  // context size the model was trained on
  uint32_t n_embd;
  uint32_t n_head;
  uint32_t n_head_kv;
  uint32_t n_layer;
  uint32_t n_rot;
  uint32_t n_ff;

  float f_norm_eps;
  float f_norm_rms_eps;

  float rope_freq_base_train;
  float rope_freq_scale_train;

  bool operator!=(const llama_hparams& other) const {
    return static_cast<bool>(memcmp(this, &other, sizeof(llama_hparams)));  // NOLINT
  }

  uint32_t n_gqa() const { return n_head / n_head_kv; }

  uint32_t n_embd_head() const { return n_embd / n_head; }

  uint32_t n_embd_gqa() const { return n_embd / n_gqa(); }
};

struct llama_vocab {
  using id = int32_t;
  using token = std::string;
  using ttype = llama_token_type;

  struct token_data {
    token text;
    float score;
    ttype type;
  };

  enum llama_vocab_type type = LLAMA_VOCAB_TYPE_SPM;

  std::unordered_map<token, id> token_to_id;
  std::vector<token_data> id_to_token;

  std::map<std::pair<std::string, std::string>, int> bpe_ranks;

  // default LLaMA special tokens
  id special_bos_id = 1;
  id special_eos_id = 2;
  id special_unk_id = 0;
  id special_sep_id = -1;
  id special_pad_id = -1;

  id linefeed_id = 13;

  int find_bpe_rank(std::string token_left, std::string token_right) const {
    replace_all(token_left, " ", "\u0120");
    replace_all(token_left, "\n", "\u010A");
    replace_all(token_right, " ", "\u0120");
    replace_all(token_right, "\n", "\u010A");

    auto it = bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == bpe_ranks.end()) {
      return -1;
    }

    return it->second;
  }
};

struct llama_layer {
  // normalization
  struct ggml_tensor* attn_norm;
  struct ggml_tensor* attn_norm_b;
  struct ggml_tensor* attn_norm_2;
  struct ggml_tensor* attn_norm_2_b;

  // attention
  struct ggml_tensor* wq;
  struct ggml_tensor* wk;
  struct ggml_tensor* wv;
  struct ggml_tensor* wo;
  struct ggml_tensor* wqkv;

  // attention bias
  struct ggml_tensor* bo;
  struct ggml_tensor* bqkv;

  // normalization
  struct ggml_tensor* ffn_norm;
  struct ggml_tensor* ffn_norm_b;

  // ff
  struct ggml_tensor* w1;  // ffn_gate
  struct ggml_tensor* w2;  // ffn_down
  struct ggml_tensor* w3;  // ffn_up

  // ff bias
  struct ggml_tensor* b2;  // ffn_down
  struct ggml_tensor* b3;  // ffn_up
};

struct llama_buffer {
  void* data = NULL;
  size_t size = 0;

  // fallback to malloc / free
  // useful in cases where CUDA can try to allocate PINNED memory
  bool fallback = false;

  void resize(size_t n) {
    llama_host_free(data);

    data = llama_host_malloc(n);
    if (!data) {
      fallback = true;
      data = malloc(n);
    } else {
      fallback = false;
    }

    GGML_ASSERT(data);
    size = n;
  }

  ~llama_buffer() {
    if (data) {
      if (fallback) {  // NOLINT
        free(data);
      } else {
        llama_host_free(data);
      }
    }

    data = NULL;
  }
};

struct llama_file {
  // use FILE * so we don't have to re-open the file to mmap
  FILE* fp;
  size_t size;

  llama_file(const char* fname, const char* mode) {
    fp = std::fopen(fname, mode);
    if (fp == NULL) {
      throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
    }
    seek(0, SEEK_END);
    size = tell();
    seek(0, SEEK_SET);
  }

  size_t tell() const {
    long ret = std::ftell(fp);
    GGML_ASSERT(ret != -1);  // this really shouldn't fail
    return (size_t)ret;
  }

  void seek(size_t offset, int whence) const {
    int ret = std::fseek(fp, (long)offset, whence);
    GGML_ASSERT(ret == 0);  // same
  }

  void read_raw(void* ptr, size_t len) const {
    if (len == 0) {
      return;
    }
    errno = 0;
    std::size_t ret = std::fread(ptr, len, 1, fp);
    if (ferror(fp)) {
      throw std::runtime_error(format("read error: %s", strerror(errno)));
    }
    if (ret != 1) {
      throw std::runtime_error(std::string("unexpectedly reached end of file"));
    }
  }

  uint32_t read_u32() const {
    uint32_t ret;
    read_raw(&ret, sizeof(ret));
    return ret;
  }

  void write_raw(const void* ptr, size_t len) const {
    if (len == 0) {
      return;
    }
    errno = 0;
    size_t ret = std::fwrite(ptr, len, 1, fp);
    if (ret != 1) {
      throw std::runtime_error(format("write error: %s", strerror(errno)));
    }
  }

  void write_u32(std::uint32_t val) const { write_raw(&val, sizeof(val)); }

  ~llama_file() {
    if (fp) {
      std::fclose(fp);
    }
  }
};

struct llama_mmap {
  void* addr;
  size_t size;

  llama_mmap(const llama_mmap&) = delete;

  static constexpr bool SUPPORTED = true;

  llama_mmap(struct llama_file* file, size_t prefetch = (size_t)-1 /* -1 = max value */, bool numa = false) {
    size = file->size;
    int fd = fileno(file->fp);
    int flags = MAP_SHARED;

    // prefetch/readahead impairs performance on NUMA systems
    if (numa) {
      prefetch = 0;
    }

    if (prefetch) {
      flags |= MAP_POPULATE;
    }

    addr = mmap(NULL, file->size, PROT_READ, flags, fd, 0);
    if (addr == MAP_FAILED) {
      throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
    }

    if (prefetch > 0) {
      // Advise the kernel to preload the mapped memory
      if (posix_madvise(addr, std::min(file->size, prefetch), POSIX_MADV_WILLNEED)) {
        fprintf(stderr, "warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n", strerror(errno));
      }
    }

    if (numa) {
      // advise the kernel not to use readahead
      // (because the next page might not belong on the same node)
      if (posix_madvise(addr, file->size, POSIX_MADV_RANDOM)) {
        fprintf(stderr, "warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n", strerror(errno));
      }
    }
  }

  ~llama_mmap() { munmap(addr, size); }
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct llama_mlock {
  void* addr = NULL;
  size_t size = 0;

  bool failed_already = false;

  llama_mlock() {}
  llama_mlock(const llama_mlock&) = delete;

  ~llama_mlock() {
    if (size) {
      raw_unlock(addr, size);
    }
  }

  void init(void* ptr) {
    GGML_ASSERT(addr == NULL && size == 0);  // NOLINT
    addr = ptr;
  }

  void grow_to(size_t target_size) {
    GGML_ASSERT(addr);
    if (failed_already) {
      return;
    }
    size_t granularity = lock_granularity();
    target_size = (target_size + granularity - 1) & ~(granularity - 1);
    if (target_size > size) {
      if (raw_lock((uint8_t*)addr + size, target_size - size)) {
        size = target_size;
      } else {
        failed_already = true;
      }
    }
  }

#ifdef _POSIX_MEMLOCK_RANGE
  static constexpr bool SUPPORTED = true;

  static size_t lock_granularity() { return (size_t)sysconf(_SC_PAGESIZE); }

#ifdef __APPLE__
#define MLOCK_SUGGESTION                                              \
  "Try increasing the sysctl values 'vm.user_wire_limit' and "        \
  "'vm.global_user_wire_limit' and/or "                               \
  "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing " \
  "RLIMIT_MLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION "Try increasing RLIMIT_MLOCK ('ulimit -l' as root).\n"
#endif

  bool raw_lock(const void* addr, size_t size) const {
    if (!mlock(addr, size)) {
      return true;
    }

    char* errmsg = std::strerror(errno);
    bool suggest = (errno == ENOMEM);

    // Check if the resource limit is fine after all
    struct rlimit lock_limit;
    if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
      suggest = false;
    }
    if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
      suggest = false;
    }

    fprintf(stderr,
            "warning: failed to mlock %zu-byte buffer (after previously "
            "locking %zu bytes): %s\n%s",
            size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
    return false;
  }

#undef MLOCK_SUGGESTION

  static void raw_unlock(void* addr, size_t size) {
    if (munlock(addr, size)) {
      fprintf(stderr, "warning: failed to munlock buffer: %s\n", std::strerror(errno));
    }
  }

#else
  static constexpr bool SUPPORTED = false;

  static size_t lock_granularity() { return (size_t)65536; }

  bool raw_lock(const void* addr, size_t len) const {
    fprintf(stderr, "warning: mlock not supported on this system\n");
    return false;
  }

  static void raw_unlock(const void* addr, size_t len) {}
#endif
};

struct llama_model {
  e_model type = MODEL_UNKNOWN;
  llm_arch arch = LLM_ARCH_UNKNOWN;
  llama_ftype ftype = LLAMA_FTYPE_ALL_F32;

  std::string name = "n/a";

  llama_hparams hparams = {};
  llama_vocab vocab;

  struct ggml_tensor* tok_embeddings;
  struct ggml_tensor* pos_embeddings;

  struct ggml_tensor* output_norm;
  struct ggml_tensor* output_norm_b;
  struct ggml_tensor* output;

  std::vector<llama_layer> layers;

  int n_gpu_layers;

  // context
  struct ggml_context* ctx = NULL;

  // the model memory buffer
  llama_buffer buf;

  // model memory mapped file
  std::unique_ptr<llama_mmap> mapping;

  // objects representing data potentially being locked in memory
  llama_mlock mlock_buf;
  llama_mlock mlock_mmap;

  // for quantize-stats only
  std::vector<std::pair<std::string, struct ggml_tensor*>> tensors_by_name;

  int64_t t_load_us = 0;
  int64_t t_start_us = 0;

  ~llama_model() {
    if (ctx) {
      ggml_free(ctx);
    }

#ifdef GGML_USE_CUBLAS
    for (size_t i = 0; i < tensors_by_name.size(); ++i) {
      ggml_cuda_free_data(tensors_by_name[i].second);
    }
    ggml_cuda_free_scratch();
#elif defined(GGML_USE_CLBLAST)
    for (size_t i = 0; i < tensors_by_name.size(); ++i) {
      ggml_cl_free_data(tensors_by_name[i].second);
    }
#endif
  }
};

struct llama_model_params llama_model_default_params() {
  struct llama_model_params result = {
      /*.n_gpu_layers                =*/0,
      /*.main_gpu                    =*/0,
      /*.tensor_split                =*/nullptr,
      /*.progress_callback           =*/nullptr,
      /*.progress_callback_user_data =*/nullptr,
      /*.vocab_only                  =*/false,
      /*.use_mmap                    =*/true,
      /*.use_mlock                   =*/false,
  };

  return result;
}

static void llama_log_callback_default(ggml_log_level level, const char* text, void* user_data) {
  (void)level;
  (void)user_data;
  fputs(text, stderr);
  fflush(stderr);
}

struct llama_state {
  // We save the log callback globally
  ggml_log_callback log_callback = llama_log_callback_default;
  void* log_callback_user_data = nullptr;
};

static llama_state g_llama_state;

static void llama_log_internal_v(ggml_log_level level, const char* format, va_list args) {
  va_list args_copy;
  va_copy(args_copy, args);
  char buffer[128];
  int len = vsnprintf(buffer, 128, format, args);
  if (len < 128) {
    g_llama_state.log_callback(level, buffer, g_llama_state.log_callback_user_data);
  } else {
    char* buffer2 = new char[len + 1];
    vsnprintf(buffer2, len + 1, format, args_copy);
    buffer2[len] = 0;
    g_llama_state.log_callback(level, buffer2, g_llama_state.log_callback_user_data);
    delete[] buffer2;
  }
  va_end(args_copy);
}

static void llama_log_internal(ggml_log_level level, const char* format, ...) {
  va_list args;
  va_start(args, format);
  llama_log_internal_v(level, format, args);
  va_end(args);
}

static const char* llama_model_type_name(e_model type) {
  switch (type) {
    case MODEL_1B:
      return "1B";
    case MODEL_3B:
      return "3B";
    case MODEL_7B:
      return "7B";
    case MODEL_13B:
      return "13B";
    case MODEL_15B:
      return "15B";
    case MODEL_30B:
      return "30B";
    case MODEL_34B:
      return "34B";
    case MODEL_40B:
      return "40B";
    case MODEL_65B:
      return "65B";
    case MODEL_70B:
      return "70B";
    default:
      return "?B";
  }
}

static std::string llama_model_ftype_name(llama_ftype ftype) {
  if (ftype & LLAMA_FTYPE_GUESSED) {
    return llama_model_ftype_name((enum llama_ftype)(ftype & ~LLAMA_FTYPE_GUESSED)) + " (guessed)";
  }

  switch (ftype) {
    case LLAMA_FTYPE_ALL_F32:
      return "all F32";
    case LLAMA_FTYPE_MOSTLY_F16:
      return "mostly F16";
    case LLAMA_FTYPE_MOSTLY_Q4_0:
      return "mostly Q4_0";
    case LLAMA_FTYPE_MOSTLY_Q4_1:
      return "mostly Q4_1";
    case LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16:
      return "mostly Q4_1, some F16";
    case LLAMA_FTYPE_MOSTLY_Q5_0:
      return "mostly Q5_0";
    case LLAMA_FTYPE_MOSTLY_Q5_1:
      return "mostly Q5_1";
    case LLAMA_FTYPE_MOSTLY_Q8_0:
      return "mostly Q8_0";

    // K-quants
    case LLAMA_FTYPE_MOSTLY_Q2_K:
      return "mostly Q2_K";
    case LLAMA_FTYPE_MOSTLY_Q3_K_S:
      return "mostly Q3_K - Small";
    case LLAMA_FTYPE_MOSTLY_Q3_K_M:
      return "mostly Q3_K - Medium";
    case LLAMA_FTYPE_MOSTLY_Q3_K_L:
      return "mostly Q3_K - Large";
    case LLAMA_FTYPE_MOSTLY_Q4_K_S:
      return "mostly Q4_K - Small";
    case LLAMA_FTYPE_MOSTLY_Q4_K_M:
      return "mostly Q4_K - Medium";
    case LLAMA_FTYPE_MOSTLY_Q5_K_S:
      return "mostly Q5_K - Small";
    case LLAMA_FTYPE_MOSTLY_Q5_K_M:
      return "mostly Q5_K - Medium";
    case LLAMA_FTYPE_MOSTLY_Q6_K:
      return "mostly Q6_K";

    default:
      return "unknown, may not work";
  }
}

enum llama_fver {
  GGUF_FILE_VERSION_V1 = 1,
  GGUF_FILE_VERSION_V2 = 2,
};

static const char* llama_file_version_name(llama_fver version) {
  switch (version) {
    case GGUF_FILE_VERSION_V1:
      return "GGUF V1 (support until nov 2023)";
    case GGUF_FILE_VERSION_V2:
      return "GGUF V2 (latest)";
  }

  return "unknown";
}

LLAMA_ATTRIBUTE_FORMAT(2, 3)
static void llama_log_internal(ggml_log_level level, const char* format, ...);
static void llama_log_callback_default(ggml_log_level level, const char* text, void* user_data);

#define LLAMA_LOG_INFO(...) llama_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define LLAMA_LOG_WARN(...) llama_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

static std::string llama_format_tensor_shape(const struct ggml_tensor* t) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%5" PRId64, t->ne[0]);
  for (int i = 1; i < GGML_MAX_DIMS; i++) {
    snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %5" PRId64, t->ne[i]);
  }
  return buf;
}

struct llm_symbol {
  using index = int;
  index prev;
  index next;
  const char* text;
  size_t n;
};

// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

struct llm_bigram_bpe {
  struct comparator {
    bool operator()(const llm_bigram_bpe& l, const llm_bigram_bpe& r) const {
      return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
    }
  };

  using queue_storage = std::vector<llm_bigram_bpe>;
  using queue = std::priority_queue<llm_bigram_bpe, queue_storage, comparator>;
  llm_symbol::index left;
  llm_symbol::index right;
  std::string text;
  int rank;
  size_t size;
};

static llama_token llama_byte_to_token(const llama_vocab& vocab, uint8_t ch) {
  char buf[7];
  int result = snprintf(buf, sizeof(buf), "<0x%02X>", ch);
  GGML_ASSERT(0 <= result && result < 7);
  return vocab.token_to_id.at(buf);
}

struct llm_tokenizer_bpe {
  llm_tokenizer_bpe(const llama_vocab& vocab) : vocab(vocab) {}

  void tokenize(const std::string& text, std::vector<llama_vocab::id>& output) {
    int final_prev_index = -1;
    auto word_collection = bpe_gpt2_preprocess(text);

    symbols_final.clear();

    for (auto& word : word_collection) {
      work_queue = llm_bigram_bpe::queue();
      symbols.clear();

      int index = 0;
      size_t offset = 0;

      while (offset < word.size()) {
        llm_symbol sym;
        size_t char_len = std::min(word.size() - offset, (size_t)::utf8_len(word[offset]));
        sym.text = word.c_str() + offset;
        sym.n = 1;
        sym.n = char_len;
        offset += sym.n;
        sym.prev = index - 1;
        sym.next = offset == word.size() ? -1 : index + 1;
        index++;
        symbols.emplace_back(sym);
      }
      for (size_t i = 1; i < symbols.size(); ++i) {
        add_new_bigram(i - 1, i);
      }

      // build token(s)
      while (!work_queue.empty()) {
        auto bigram = work_queue.top();
        work_queue.pop();

        auto& left_symbol = symbols[bigram.left];
        auto& right_symbol = symbols[bigram.right];

        if (left_symbol.n == 0 || right_symbol.n == 0) {
          continue;
        }
        std::string left_token = std::string(left_symbol.text, left_symbol.n);
        std::string right_token = std::string(right_symbol.text, right_symbol.n);
        if (left_token + right_token != bigram.text) {
          continue;  // Skip this bigram if it's outdated
        }

        // merge the right sym into the left one
        left_symbol.n += right_symbol.n;
        right_symbol.n = 0;

        // remove the right sym from the chain
        left_symbol.next = right_symbol.next;
        if (right_symbol.next >= 0) {
          symbols[right_symbol.next].prev = bigram.left;
        }

        add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
        add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
      }

      // add the fnished tokens to the final list keeping correct order for next and prev
      for (auto& sym : symbols) {
        if (sym.n > 0) {
          sym.prev = final_prev_index;
          sym.next = -1;
          if (final_prev_index != -1) {
            symbols_final[final_prev_index].next = symbols_final.size();
          }
          symbols_final.emplace_back(sym);
          final_prev_index = symbols_final.size() - 1;
        }
      }
    }

    symbols = symbols_final;

    if (!symbols.empty()) {
      for (int i = 0; i != -1; i = symbols[i].next) {
        auto& symbol = symbols[i];
        if (symbol.n == 0) {
          continue;
        }

        const std::string str = std::string(symbol.text, symbol.n);
        const auto token = vocab.token_to_id.find(str);

        if (token == vocab.token_to_id.end()) {
          for (auto j = str.begin(); j != str.end(); ++j) {
            std::string byte_str(1, *j);
            auto token_multibyte = vocab.token_to_id.find(byte_str);
            if (token_multibyte == vocab.token_to_id.end()) {
              try {
                llama_token token_byte = llama_byte_to_token(vocab, *j);
                output.push_back(token_byte);
              } catch (const std::out_of_range& err) {
                fprintf(stderr, "ERROR: byte not found in vocab: '%s'\n", byte_str.c_str());
              }
            } else {
              output.push_back((*token_multibyte).second);
            }
          }
        } else {
          output.push_back((*token).second);
        }
      }
    }
  }

 private:
  void add_new_bigram(int left, int right) {
    if (left == -1 || right == -1) {
      return;
    }

    std::string left_token = std::string(symbols[left].text, symbols[left].n);
    std::string right_token = std::string(symbols[right].text, symbols[right].n);

    int rank_found = -1;

    rank_found = vocab.find_bpe_rank(left_token, right_token);

    if (rank_found < 0) {
      return;
    }

    llm_bigram_bpe bigram;

    bigram.left = left;
    bigram.right = right;
    bigram.text = left_token + right_token;
    bigram.size = left_token.size() + right_token.size();
    bigram.rank = rank_found;

    work_queue.push(bigram);
  }

  // probably not 100% correct
  static std::vector<std::string> bpe_gpt2_preprocess(const std::string& text) {
    std::vector<std::string> words;

    // ref: https://github.com/openai/gpt-2/blob/a74da5d99abaaba920de8131d64da2862a8f213b/src/encoder.py#L53
    const std::string pattern =
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    const std::regex re(pattern);

    auto words_begin = std::sregex_iterator(text.begin(), text.end(), re);
    auto words_end = std::sregex_iterator();
    auto n_words = std::distance(words_begin, words_end);
    words.reserve(n_words);
    for (auto it = words_begin; it != words_end; ++it) {
      words.push_back(it->str());
    }
    return words;
  }

  const llama_vocab& vocab;

  std::vector<llm_symbol> symbols;
  std::vector<llm_symbol> symbols_final;

  llm_bigram_bpe::queue work_queue;
};

enum llm_tensor {
  LLM_TENSOR_TOKEN_EMBD,
  LLM_TENSOR_POS_EMBD,
  LLM_TENSOR_OUTPUT,
  LLM_TENSOR_OUTPUT_NORM,
  LLM_TENSOR_ROPE_FREQS,
  LLM_TENSOR_ATTN_Q,
  LLM_TENSOR_ATTN_K,
  LLM_TENSOR_ATTN_V,
  LLM_TENSOR_ATTN_QKV,
  LLM_TENSOR_ATTN_OUT,
  LLM_TENSOR_ATTN_NORM,
  LLM_TENSOR_ATTN_NORM_2,
  LLM_TENSOR_ATTN_ROT_EMBD,
  LLM_TENSOR_FFN_GATE,
  LLM_TENSOR_FFN_DOWN,
  LLM_TENSOR_FFN_UP,
  LLM_TENSOR_FFN_NORM,
};

static std::map<llm_tensor, std::string> LLAMA_TENSOR_NAMES = {
    {LLM_TENSOR_TOKEN_EMBD, "token_embd"},
    {LLM_TENSOR_OUTPUT_NORM, "output_norm"},
    {LLM_TENSOR_OUTPUT, "output"},
    {LLM_TENSOR_ROPE_FREQS, "rope_freqs"},
    {LLM_TENSOR_ATTN_NORM, "blk.%d.attn_norm"},
    {LLM_TENSOR_ATTN_Q, "blk.%d.attn_q"},
    {LLM_TENSOR_ATTN_K, "blk.%d.attn_k"},
    {LLM_TENSOR_ATTN_V, "blk.%d.attn_v"},
    {LLM_TENSOR_ATTN_OUT, "blk.%d.attn_output"},
    {LLM_TENSOR_ATTN_ROT_EMBD, "blk.%d.attn_rot_embd"},
    {LLM_TENSOR_FFN_NORM, "blk.%d.ffn_norm"},
    {LLM_TENSOR_FFN_GATE, "blk.%d.ffn_gate"},
    {LLM_TENSOR_FFN_DOWN, "blk.%d.ffn_down"},
    {LLM_TENSOR_FFN_UP, "blk.%d.ffn_up"},
};

// =======================================================================
// Main
// =======================================================================
int main_gpu = 0;

#ifdef GGML_USE_CUBLAS
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
LLAMA_LOG_INFO("%s: using " GGML_CUDA_NAME " for GPU acceleration\n", __func__);
ggml_cuda_set_main_device(main_gpu);
#define LLAMA_BACKEND_OFFLOAD GGML_BACKEND_GPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_GPU_SPLIT
#else
#define LLAMA_BACKEND_OFFLOAD GGML_BACKEND_CPU
#define LLAMA_BACKEND_OFFLOAD_SPLIT GGML_BACKEND_CPU
#endif

static const char* model_file_path = "models/llama-2-13b-chat.Q4_0.gguf";

// params
bool use_mmap = true;
bool use_mlock = true;
int n_gpu_layers = 0;
float* tensor_split = nullptr;

FILE* file = nullptr;
uint32_t magic = 0;
size_t offset = 0;
size_t nitems = 0;
size_t ret = 0;
bool ok = true;

llama_model* model = new llama_model();
llama_ftype ftype;
std::string model_arch_name;
int64_t n_elements = 0;
size_t n_bytes = 0;
size_t n_tensors = 0;
size_t n_created = 0;

struct gguf_context* gguf_ctx = (struct gguf_context*)ggml_aligned_malloc(sizeof(struct gguf_context));

int main() {
  init_type_traits();

#define P_BUILD_GGUF_CONTEXT
#define __OPEN_MODEL_FILE
  // Open model file
  {
    printf("== Open model file ==\n");
    file = fopen(model_file_path, "rb");
    if (!file) {
      printf("Could not read %s\n", model_file_path);
    }
    printf("OK\n");
    printf("\n");
  }

  // Read magic number
  {
    printf("== Read magic number ==\n");
    nitems = sizeof(magic);
    fread(&magic, 1, nitems, file);
    offset += nitems;
    printf("Magic number = 0x%x\n", magic);
    printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
    printf("\n");
  }

  // Read header
  {
    printf("== Read header ==\n");
    gguf_ctx->header.magic = magic;

    // version
    nitems = sizeof(gguf_ctx->header.version);
    fread(&gguf_ctx->header.version, 1, nitems, file);
    offset += nitems;
    printf("header.version = %d\n", gguf_ctx->header.version);
    printf("Current offset = %d; current file offset = %d\n", offset, ftell(file));
    printf("\n");

    // n_tensors
    nitems = sizeof(gguf_ctx->header.n_tensors);
    fread(&gguf_ctx->header.n_tensors, 1, nitems, file);
    offset += nitems;
    printf("header.n_tensors = %d\n", gguf_ctx->header.n_tensors);
    printf("Current offset = %d; current file offset = %d\n", offset, ftell(file));
    printf("\n");

    // n_kv
    nitems = sizeof(gguf_ctx->header.n_kv);
    fread(&gguf_ctx->header.n_kv, 1, nitems, file);
    offset += nitems;
    printf("header.n_kv = %d\n", gguf_ctx->header.n_kv);
    printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
    printf("\n");
  }

  // Read the kv pairs
  {
    printf("== Read the kv pairs ==\n");
    size_t kv_buf_size = gguf_ctx->header.n_kv * sizeof(struct gguf_kv);
    void* kv_ptr = malloc(kv_buf_size);
    gguf_ctx->kv = (struct gguf_kv*)kv_ptr;

    for (uint32_t i = 0; i < gguf_ctx->header.n_kv; ++i) {
      struct gguf_kv* kv = &gguf_ctx->kv[i];

      nitems = sizeof(kv->key.n);
      ret = fread(&kv->key.n, 1, nitems, file);
      offset += nitems;
      printf("Key length = %d\n", kv->key.n);

      kv->key.data = (char*)calloc(kv->key.n + 1, 1);
      nitems = kv->key.n;
      ret = fread(kv->key.data, 1, nitems, file);
      offset += nitems;
      printf("Key = %s\n", kv->key.data);

      nitems = sizeof(kv->type);
      fread(&kv->type, 1, nitems, file);
      offset += nitems;
      printf("Type of value = %d\n", kv->type);

      switch (kv->type) {
        case GGUF_TYPE_UINT8: {
          nitems = sizeof(kv->value.uint8);
          fread(&kv->value.uint8, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.uint8);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_INT8: {
          nitems = sizeof(kv->value.int8);
          fread(&kv->value.int8, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.int8);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_UINT16: {
          nitems = sizeof(kv->value.uint16);
          fread(&kv->value.uint16, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.uint16);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_INT16: {
          nitems = sizeof(kv->value.int16);
          fread(&kv->value.int16, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.int16);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_UINT32: {
          nitems = sizeof(kv->value.uint32);
          fread(&kv->value.uint32, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.uint32);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_INT32: {
          nitems = sizeof(kv->value.int32);
          fread(&kv->value.int32, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.int32);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_FLOAT32: {
          nitems = sizeof(kv->value.float32);
          fread(&kv->value.float32, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.float32);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_UINT64: {
          nitems = sizeof(kv->value.uint64);
          fread(&kv->value.uint64, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.uint64);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_INT64: {
          nitems = sizeof(kv->value.int64);
          fread(&kv->value.int64, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.int64);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_FLOAT64: {
          nitems = sizeof(kv->value.float64);
          fread(&kv->value.float64, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.float64);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_BOOL: {
          nitems = sizeof(kv->value.bool_);
          fread(&kv->value.bool_, 1, nitems, file);
          offset += nitems;
          printf("Value = %d\n", kv->value.bool_);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_STRING: {
          nitems = sizeof(kv->value.str.n);
          fread(&kv->value.str.n, 1, nitems, file);
          offset += nitems;

          kv->value.str.data = (char*)calloc(kv->value.str.n + 1, 1);
          nitems = kv->value.str.n;
          fread(kv->value.str.data, 1, nitems, file);
          offset += nitems;

          printf("Value = %s\n", kv->value.str.data);
          printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
          break;
        }
        case GGUF_TYPE_ARRAY: {
          nitems = sizeof(kv->value.arr.type);
          fread(&kv->value.arr.type, 1, nitems, file);
          offset += nitems;
          nitems = sizeof(kv->value.arr.n);
          fread(&kv->value.arr.n, 1, nitems, file);
          offset += nitems;

          switch (kv->value.arr.type) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
            case GGUF_TYPE_BOOL: {
              nitems = kv->value.arr.n * GGUF_TYPE_SIZE[kv->value.arr.type];
              kv->value.arr.data = malloc(nitems);
              fread(kv->value.arr.data, 1, nitems, file);
              offset += nitems;

              printf("Value = %p\n", kv->value.arr.data);
              printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));

              break;
            }
            case GGUF_TYPE_STRING: {
              size_t buf_size = kv->value.arr.n * sizeof(struct gguf_str);
              kv->value.arr.data = malloc(buf_size);
              struct gguf_str* arr = (struct gguf_str*)kv->value.arr.data;
              for (uint32_t j = 0; j < kv->value.arr.n; ++j) {
                struct gguf_str* p = &arr[j];
                p->n = 0;
                p->data = nullptr;

                nitems = sizeof(p->n);
                fread(&p->n, 1, nitems, file);
                offset += nitems;

                p->data = (char*)calloc(p->n + 1, 1);
                nitems = p->n;
                fread(p->data, 1, nitems, file);
                offset += nitems;
              }

              printf("Value = [");
              size_t i = 0;
              for (; i < kv->value.arr.n; i++) {
                if (i == 0) {
                  printf("\"%s\"", arr[i].data);
                } else if (i == 999) {
                  printf(", ...");
                  break;
                } else {
                  printf(", \"%s\"", arr[i].data);
                }
              }
              if (i < kv->value.arr.n - 1) {
                printf(", \"%s\"", arr[kv->value.arr.n - 1].data);
              }
              printf("]\n");
              printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));

              break;
            }
            case GGUF_TYPE_ARRAY:
            case GGUF_TYPE_COUNT:
              GGML_ASSERT(false && "invalid type");
              break;
          }

          break;
        }
        case GGUF_TYPE_COUNT:
          GGML_ASSERT(false && "invalid type");
      }

      printf("\n");
      fflush(stdout);
    }
  }

  // Read the tensor infos
  {
    printf("Read the tensor info\n");
    gguf_ctx->infos = (struct gguf_tensor_info*)malloc(gguf_ctx->header.n_tensors * sizeof(struct gguf_tensor_info));
    for (uint32_t i = 0; i < gguf_ctx->header.n_tensors; ++i) {
      struct gguf_tensor_info* info = &gguf_ctx->infos[i];

      for (int j = 0; j < GGML_MAX_DIMS; ++j) {
        info->ne[j] = 1;
      }

      nitems = sizeof(info->name.n);
      fread(&info->name.n, 1, nitems, file);
      offset += nitems;

      info->name.data = (char*)calloc(info->name.n + 1, 1);
      nitems = info->name.n;
      fread(info->name.data, 1, nitems, file);
      offset += nitems;
      printf("Tensor name = %s\n", info->name.data);

      nitems = sizeof(info->n_dims);
      fread(&info->n_dims, 1, nitems, file);
      offset += nitems;
      printf("Tensor n_dims = %d\n", info->n_dims);

      printf("Tensor dims = [");
      for (uint32_t j = 0; j < info->n_dims; ++j) {
        nitems = sizeof(info->ne[j]);
        fread(&info->ne[j], 1, nitems, file);
        offset += nitems;
        if (j == 0) {
          printf("%d", info->ne[j]);
        } else {
          printf(", %d", info->ne[j]);
        }
      }
      printf("]\n");

      nitems = sizeof(info->type);
      fread(&info->type, 1, nitems, file);
      offset += nitems;
      printf("Tensor data type = %d\n", info->type);

      nitems = sizeof(info->offset);
      fread(&info->offset, 1, nitems, file);
      offset += nitems;
      printf("Tensor offset = %lld\n", info->offset);
      printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
      printf("\n");
    }
  }

  // Read alignment
  {
    gguf_ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
    int alignment_idx = -1;
    const char* alignment_key = "general.alignment";
    for (size_t i = 0; i < gguf_ctx->header.n_kv; i++) {
      if (strcmp(alignment_key, gguf_ctx->kv[i].key.data) == 0) {
        alignment_idx = i;
      }
    }

    if (alignment_idx != -1) {
      GGML_ASSERT(gguf_ctx->kv[alignment_idx].type == GGUF_TYPE_UINT32);
      gguf_ctx->alignment = gguf_ctx->kv[alignment_idx].value.uint32;
    }
  }

  // We require the data section to be aligned, so take into account any
  // padding
  {
    const size_t offset_pad = offset % gguf_ctx->alignment;

    if (offset_pad != 0) {
      offset += gguf_ctx->alignment - offset_pad;
      fseek(file, offset, SEEK_SET);
      printf("Current offset = %d; actual file offset = %d\n", offset, ftell(file));
    }

    // Store the current file offset - this is where the data section starts
    gguf_ctx->offset = offset;
  }

  // Compute the total size of the data section, taking into account the
  // alignment
  {
    gguf_ctx->size = 0;
    for (uint32_t i = 0; i < gguf_ctx->header.n_tensors; ++i) {
      struct gguf_tensor_info* info = &gguf_ctx->infos[i];

      const int64_t element_count =
          (int64_t)info->ne[0] * (int64_t)info->ne[1] * (int64_t)info->ne[2] * (int64_t)info->ne[3];

      if (element_count % type_traits[info->type].blck_size != 0) {
        fprintf(stderr,
                "%s: tensor '%s' number of elements %lld is not a multiple of "
                "block size (%d)\n",
                __func__, info->name.data, element_count, type_traits[info->type].blck_size);
        fclose(file);
        gguf_free(gguf_ctx);
        return 1;
      }

      const size_t size_cur = (element_count * type_traits[info->type].type_size) / type_traits[info->type].blck_size;
      gguf_ctx->size += GGML_PAD(size_cur, gguf_ctx->alignment);
    }
  }

  // Load the tensor data only if requested
  {
    // If the provided gguf_context is no_alloc, then we create "empty"
    // tensors and do not read the binary blob otherwise, we load the binary
    // blob into the created ggml_context as well, and point the "data"
    // members of the ggml_tensor structs to the appropriate locations in the
    // binary blob

    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &ctx_meta,
    };

    // compute the exact size needed for the new ggml_context
    const size_t mem_size =
        params.no_alloc ? (gguf_ctx->header.n_tensors) * (GGML_OBJECT_SIZE + GGML_TENSOR_SIZE)
                        : (gguf_ctx->header.n_tensors + 1) * (GGML_OBJECT_SIZE + GGML_TENSOR_SIZE) + gguf_ctx->size;

    struct ggml_init_params pdata = {
        .mem_size = mem_size,
        .mem_buffer = NULL,
        .no_alloc = params.no_alloc,
    };

    ggml_context* tmp_ctx = ggml_init(pdata);
    *params.ctx = tmp_ctx;

    // ggml_set_no_alloc(ctx_meta, true);
    ctx_meta->no_alloc = true;

    // create the tensors
    for (uint32_t i = 0; i < gguf_ctx->header.n_tensors; ++i) {
      const int64_t ne[GGML_MAX_DIMS] = {
          (int64_t)gguf_ctx->infos[i].ne[0],
          (int64_t)gguf_ctx->infos[i].ne[1],
          (int64_t)gguf_ctx->infos[i].ne[2],
          (int64_t)gguf_ctx->infos[i].ne[3],
      };

      struct ggml_tensor* cur = ggml_new_tensor(ctx_meta, gguf_ctx->infos[i].type, gguf_ctx->infos[i].n_dims, ne);
      if (cur != nullptr) {
        ggml_set_name(cur, gguf_ctx->infos[i].name.data);
      } else {
        throw std::runtime_error(format("Could not create tensor %s", gguf_ctx->infos[i].name.data));
      }
    }
  }

  // Print metadata
  {
    for (int i = 0; i < gguf_ctx->header.n_tensors; i++) {
      const char* name = gguf_ctx->infos[i].name.data;
      struct ggml_tensor* t = ggml_get_tensor(ctx_meta, name);
      n_elements += (t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3]);
      n_bytes += ggml_nbytes(t);
    }

    LLAMA_LOG_INFO(
        "%s: loaded meta data with %d key-value pairs and %d tensors from %s "
        "(version %s)\n\n",
        __func__, gguf_ctx->header.n_kv, gguf_ctx->header.n_tensors, model_file_path,
        llama_file_version_name((llama_fver)gguf_ctx->header.version));

    {
      std::map<enum ggml_type, uint32_t> n_type;

      uint32_t n_type_max = 0;
      enum ggml_type type_max = GGML_TYPE_F32;

      for (int i = 0; i < gguf_ctx->header.n_tensors; i++) {
        const char* name = gguf_ctx->infos[i].name.data;
        struct ggml_tensor* tensor = ggml_get_tensor(ctx_meta, name);

        n_type[tensor->type]++;

        if (n_type_max < n_type[tensor->type]) {
          n_type_max = n_type[tensor->type];
          type_max = tensor->type;
        }

        LLAMA_LOG_INFO("%s: - tensor %4d: %32s %-8s [ %s ]\n", __func__, i, name, type_traits[tensor->type].type_name,
                       llama_format_tensor_shape(tensor).c_str());
      }

      printf("\n");

      switch (type_max) {
        case GGML_TYPE_F32:
          ftype = LLAMA_FTYPE_ALL_F32;
          break;
        case GGML_TYPE_F16:
          ftype = LLAMA_FTYPE_MOSTLY_F16;
          break;
        case GGML_TYPE_Q4_0:
          ftype = LLAMA_FTYPE_MOSTLY_Q4_0;
          break;
        case GGML_TYPE_Q4_1:
          ftype = LLAMA_FTYPE_MOSTLY_Q4_1;
          break;
        case GGML_TYPE_Q5_0:
          ftype = LLAMA_FTYPE_MOSTLY_Q5_0;
          break;
        case GGML_TYPE_Q5_1:
          ftype = LLAMA_FTYPE_MOSTLY_Q5_1;
          break;
        case GGML_TYPE_Q8_0:
          ftype = LLAMA_FTYPE_MOSTLY_Q8_0;
          break;
        case GGML_TYPE_Q2_K:
          ftype = LLAMA_FTYPE_MOSTLY_Q2_K;
          break;
        case GGML_TYPE_Q3_K:
          ftype = LLAMA_FTYPE_MOSTLY_Q3_K_M;
          break;
        case GGML_TYPE_Q4_K:
          ftype = LLAMA_FTYPE_MOSTLY_Q4_K_M;
          break;
        case GGML_TYPE_Q5_K:
          ftype = LLAMA_FTYPE_MOSTLY_Q5_K_M;
          break;
        case GGML_TYPE_Q6_K:
          ftype = LLAMA_FTYPE_MOSTLY_Q6_K;
          break;
        default: {
          LLAMA_LOG_WARN("%s: unknown type %s\n", __func__, type_traits[type_max].type_name);
          ftype = LLAMA_FTYPE_ALL_F32;

        } break;
      }

      // this is a way to mark that we have "guessed" the file type
      ftype = (llama_ftype)(ftype | LLAMA_FTYPE_GUESSED);
      {
        const int kid = gguf_find_key(gguf_ctx, "general.file_type");
        if (kid >= 0) {
          GGML_ASSERT(gguf_ctx->kv[kid].type == GGUF_TYPE_UINT32);
          ftype = (llama_ftype)gguf_ctx->kv[kid].value.uint32;
        }
      }

      for (uint64_t i = 0; i < gguf_ctx->header.n_kv; i++) {
        const char* name = gguf_ctx->kv[i].key.data;
        const enum gguf_type type = gguf_ctx->kv[i].type;

        LLAMA_LOG_INFO("%s: - kv %3d: %42s %-8s\n", __func__, i, name, GGUF_TYPE_NAME[type]);
      }

      printf("\n");

      // print type counts
      for (auto& kv : n_type) {
        if (kv.second == 0) {
          continue;
        }

        LLAMA_LOG_INFO("%s: - type %4s: %4d tensors\n", __func__, type_traits[kv.first].type_name, kv.second);
      }
    }
  }

#define __CLOSE_MODEL_FILE
  fclose(file);

  // Save number of tensors
  n_tensors = gguf_ctx->header.n_tensors;
  printf("\n");

#define P_BUILD_MODEL_OBJECT
  {
    // Model name
    {
      const std::string model_name_key = LLM_KV_NAMES[LLM_KV_GENERAL_NAME];
      const int kid = gguf_find_key(gguf_ctx, model_name_key.c_str());
      model->name = gguf_ctx->kv[kid].value.str.data;
      LLAMA_LOG_INFO("%s: - %42s %-20s\n", __func__, "model->name:", model->name.c_str());
    }

    // Model arch
    {
      const std::string arch_key = LLM_KV_NAMES[LLM_KV_GENERAL_ARCHITECTURE];
      const int kid = gguf_find_key(gguf_ctx, arch_key.c_str());
      const std::string arch_value = gguf_ctx->kv[kid].value.str.data;
      model->arch = LLM_ARCH_NAMES[arch_value];
      model_arch_name = arch_value;
      LLAMA_LOG_INFO("%s: - %42s %-20s\n", __func__, "model->arch:", arch_value.c_str());
    }

    // Model hyperparameters
    {
      bool vocab_only = false;
      auto& hparams = model->hparams;

      hparams.vocab_only = vocab_only;
      LLAMA_LOG_INFO("%s: - %42s %-20s\n", __func__,
                     "model->hparams.vocab_only:", hparams.vocab_only ? "true" : "false");

      // get hparams kv
      std::string key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_LIST], model_arch_name);
      int kid = gguf_find_key(gguf_ctx, key.c_str());
      hparams.n_vocab = gguf_ctx->kv[kid].value.arr.n;
      LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_vocab:", hparams.n_vocab);

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_CONTEXT_LENGTH], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      hparams.n_ctx_train = gguf_ctx->kv[kid].value.uint32;
      LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_ctx_train:", hparams.n_ctx_train);

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_EMBEDDING_LENGTH], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      hparams.n_embd = gguf_ctx->kv[kid].value.uint32;
      LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_embd:", hparams.n_embd);

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_FEED_FORWARD_LENGTH], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      hparams.n_ff = gguf_ctx->kv[kid].value.uint32;
      LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_ff:", hparams.n_ff);

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_ATTENTION_HEAD_COUNT], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      hparams.n_head = gguf_ctx->kv[kid].value.uint32;
      LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_head:", hparams.n_head);

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_BLOCK_COUNT], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      hparams.n_layer = gguf_ctx->kv[kid].value.uint32;
      LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_layer:", hparams.n_layer);

      // n_head_kv is optional, default to n_head
      hparams.n_head_kv = hparams.n_head;
      key = build_llm_key(LLM_KV_NAMES[LLM_KV_ATTENTION_HEAD_COUNT_KV], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid > 0) {
        hparams.n_head_kv = gguf_ctx->kv[kid].value.uint32;
      }
      LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_head_kv:", hparams.n_head_kv);

      // rope_freq_base (optional)
      hparams.rope_freq_base_train = 10000.0f;
      key = build_llm_key(LLM_KV_NAMES[LLM_KV_ROPE_FREQ_BASE], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid > 0) {
        hparams.rope_freq_base_train = gguf_ctx->kv[kid].value.float32;
      }
      LLAMA_LOG_INFO("%s: - %42s %-20.3lf\n", __func__,
                     "model->hparams.rope_freq_base_train:", hparams.rope_freq_base_train);

      // rope_freq_scale (inverse of the kv) is optional
      float ropescale = 1.0f;
      key = build_llm_key(LLM_KV_NAMES[LLM_KV_ROPE_SCALE_LINEAR], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid > 0) {
        ropescale = gguf_ctx->kv[kid].value.float32;
      }
      hparams.rope_freq_scale_train = 1.0f / ropescale;
      LLAMA_LOG_INFO("%s: - %42s %-20.3lf\n", __func__,
                     "model->hparams.rope_freq_scale_train:", hparams.rope_freq_scale_train);

      // For LLM_ARCH_LLAMA only (We already known our model arch is llama)
      {
        key = build_llm_key(LLM_KV_NAMES[LLM_KV_ATTENTION_LAYERNORM_RMS_EPS], model_arch_name);
        kid = gguf_find_key(gguf_ctx, key.c_str());
        hparams.f_norm_rms_eps = gguf_ctx->kv[kid].value.float32;
        LLAMA_LOG_INFO("%s: - %42s %-20.3lf\n", __func__, "model->hparams.f_norm_rms_eps:", hparams.f_norm_rms_eps);

        // Model type
        switch (hparams.n_layer) {
          case 26:
            model->type = e_model::MODEL_3B;
            break;
          case 32:
            model->type = e_model::MODEL_7B;
            break;
          case 40:
            model->type = e_model::MODEL_13B;
            break;
          case 48:
            model->type = e_model::MODEL_34B;
            break;
          case 60:
            model->type = e_model::MODEL_30B;
            break;
          case 80:
            model->type = hparams.n_head == hparams.n_head_kv ? e_model::MODEL_65B : e_model::MODEL_70B;
            break;
          default:
            model->type = e_model::MODEL_UNKNOWN;
        }

        LLAMA_LOG_INFO("%s: - %42s %-20d\n", __func__, "model->type:", model->type);
      }
    }

    // Model file type
    {
      model->ftype = ftype;
      LLAMA_LOG_INFO("%s: - %42s %-20d\n", __func__, "model->ftype:", model->ftype);
    }

    // Model vocab
    {
      // GGUF_GET_KEY(gguf_ctx, tokenizer_name, gguf_get_val_str, GGUF_TYPE_STRING, true, kv(LLM_KV_TOKENIZER_MODEL));
      std::string key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_MODEL], model_arch_name);
      int kid = gguf_find_key(gguf_ctx, key.c_str());
      std::string tokenizer_name = gguf_ctx->kv[kid].value.str.data;

      // It should be "llama"
      if (tokenizer_name != "llama") {
        throw std::runtime_error("The tokenizer name is NOT \"llama\".");
      }

      model->vocab.type = LLAMA_VOCAB_TYPE_SPM;

      // Default special tokens
      model->vocab.special_bos_id = 1;
      model->vocab.special_eos_id = 2;
      model->vocab.special_unk_id = 0;
      model->vocab.special_sep_id = -1;
      model->vocab.special_pad_id = -1;

      model->vocab.id_to_token.resize(model->hparams.n_vocab);

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_LIST], model_arch_name);
      const int token_idx = gguf_find_key(gguf_ctx, key.c_str());
      if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
      }

      const float* scores = nullptr;
      key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_SCORES], model_arch_name);
      const int scores_idx = gguf_find_key(gguf_ctx, key.c_str());
      if (scores_idx == -1) {
        throw std::runtime_error("cannot find tokenizer scores in model file\n");
      }
      scores = (const float*)gguf_ctx->kv[scores_idx].value.arr.data;

      const int* toktypes = nullptr;
      key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_TOKEN_TYPE], model_arch_name);
      const int toktypes_idx = gguf_find_key(gguf_ctx, key.c_str());
      if (toktypes_idx == -1) {
        throw std::runtime_error("cannot find tokenizer token types in model file\n");
      }
      toktypes = (const int*)gguf_ctx->kv[toktypes_idx].value.arr.data;

      for (uint32_t i = 0; i < model->hparams.n_vocab; i++) {
        std::string word = ((struct gguf_str*)gguf_ctx->kv[token_idx].value.arr.data)[i].data;

        model->vocab.token_to_id[word] = i;
        model->vocab.id_to_token[i].text = std::move(word);
        model->vocab.id_to_token[i].score = scores ? scores[i] : 0.0f;
        model->vocab.id_to_token[i].type = toktypes ? (llama_token_type)toktypes[i] : LLAMA_TOKEN_TYPE_NORMAL;
      }

      // Determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
      if (model->vocab.type == LLAMA_VOCAB_TYPE_SPM) {
        char buf[7];
        int result = snprintf(buf, sizeof(buf), "<0x%02X>", '\n');
        GGML_ASSERT(0 <= result && result < 7);
        model->vocab.linefeed_id = model->vocab.token_to_id.at(buf);
      } else {
        std::vector<llama_vocab::id> output;
        llm_tokenizer_bpe tokenizer(model->vocab);
        tokenizer.tokenize("\n", output);
        model->vocab.linefeed_id = output[0];
      }

      // Special tokens
      key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_BOS_ID], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid != -1) {
        model->vocab.special_bos_id = gguf_ctx->kv[kid].value.uint32;
      }

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_EOS_ID], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid != -1) {
        model->vocab.special_eos_id = gguf_ctx->kv[kid].value.uint32;
      }

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_UNK_ID], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid != -1) {
        model->vocab.special_unk_id = gguf_ctx->kv[kid].value.uint32;
      }

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_SEP_ID], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid != -1) {
        model->vocab.special_sep_id = gguf_ctx->kv[kid].value.uint32;
      }

      key = build_llm_key(LLM_KV_NAMES[LLM_KV_TOKENIZER_PAD_ID], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid != -1) {
        model->vocab.special_pad_id = gguf_ctx->kv[kid].value.uint32;
      }
    }
  }

  printf("\n");

  // Print model metadata
  {
    const auto& hparams = model->hparams;
    const auto& vocab = model->vocab;
    const llama_fver fver = (llama_fver)gguf_ctx->header.version;

    // hparams
    LLAMA_LOG_INFO("%s: format           = %s\n", __func__, llama_file_version_name(fver));
    LLAMA_LOG_INFO("%s: arch             = %s\n", __func__, model_arch_name.c_str());
    LLAMA_LOG_INFO("%s: vocab type       = %s\n", __func__,
                   vocab.type == LLAMA_VOCAB_TYPE_SPM ? "SPM" : "BPE");  // TODO: fix
    LLAMA_LOG_INFO("%s: n_vocab          = %u\n", __func__, hparams.n_vocab);
    LLAMA_LOG_INFO("%s: n_merges         = %u\n", __func__, (int)vocab.bpe_ranks.size());
    LLAMA_LOG_INFO("%s: n_ctx_train      = %u\n", __func__, hparams.n_ctx_train);
    LLAMA_LOG_INFO("%s: n_embd           = %u\n", __func__, hparams.n_embd);
    LLAMA_LOG_INFO("%s: n_head           = %u\n", __func__, hparams.n_head);
    LLAMA_LOG_INFO("%s: n_head_kv        = %u\n", __func__, hparams.n_head_kv);
    LLAMA_LOG_INFO("%s: n_layer          = %u\n", __func__, hparams.n_layer);
    LLAMA_LOG_INFO("%s: n_rot            = %u\n", __func__, hparams.n_rot);  // a.k.a. n_embd_head, n_head_dim
    LLAMA_LOG_INFO("%s: n_gqa            = %u\n", __func__, hparams.n_gqa());
    LLAMA_LOG_INFO("%s: f_norm_eps       = %.1e\n", __func__, hparams.f_norm_eps);
    LLAMA_LOG_INFO("%s: f_norm_rms_eps   = %.1e\n", __func__, hparams.f_norm_rms_eps);
    LLAMA_LOG_INFO("%s: n_ff             = %u\n", __func__, hparams.n_ff);
    LLAMA_LOG_INFO("%s: freq_base_train  = %.1f\n", __func__, hparams.rope_freq_base_train);
    LLAMA_LOG_INFO("%s: freq_scale_train = %g\n", __func__, hparams.rope_freq_scale_train);
    LLAMA_LOG_INFO("%s: model type       = %s\n", __func__, llama_model_type_name(model->type));
    LLAMA_LOG_INFO("%s: model ftype      = %s\n", __func__, llama_model_ftype_name(model->ftype).c_str());
    LLAMA_LOG_INFO("%s: model params     = %.2f B\n", __func__, n_elements * 1e-9);
    if (n_bytes < GB) {
      LLAMA_LOG_INFO("%s: model size       = %.2f MiB (%.2f BPW) \n", __func__, n_bytes / 1024.0 / 1024.0,
                     n_bytes * 8.0 / n_elements);
    } else {
      LLAMA_LOG_INFO("%s: model size       = %.2f GiB (%.2f BPW) \n", __func__, n_bytes / 1024.0 / 1024.0 / 1024.0,
                     n_bytes * 8.0 / n_elements);
    }

    // general kv
    LLAMA_LOG_INFO("%s: general.name     = %s\n", __func__, model->name.c_str());

    // special tokens
    if (vocab.special_bos_id != -1) {
      LLAMA_LOG_INFO("%s: BOS token = %d '%s'\n", __func__, vocab.special_bos_id,
                     vocab.id_to_token[vocab.special_bos_id].text.c_str());
    }
    if (vocab.special_eos_id != -1) {
      LLAMA_LOG_INFO("%s: EOS token = %d '%s'\n", __func__, vocab.special_eos_id,
                     vocab.id_to_token[vocab.special_eos_id].text.c_str());
    }
    if (vocab.special_unk_id != -1) {
      LLAMA_LOG_INFO("%s: UNK token = %d '%s'\n", __func__, vocab.special_unk_id,
                     vocab.id_to_token[vocab.special_unk_id].text.c_str());
    }
    if (vocab.special_sep_id != -1) {
      LLAMA_LOG_INFO("%s: SEP token = %d '%s'\n", __func__, vocab.special_sep_id,
                     vocab.id_to_token[vocab.special_sep_id].text.c_str());
    }
    if (vocab.special_pad_id != -1) {
      LLAMA_LOG_INFO("%s: PAD token = %d '%s'\n", __func__, vocab.special_pad_id,
                     vocab.id_to_token[vocab.special_pad_id].text.c_str());
    }
    if (vocab.linefeed_id != -1) {
      LLAMA_LOG_INFO("%s: LF token  = %d '%s'\n", __func__, vocab.linefeed_id,
                     vocab.id_to_token[vocab.linefeed_id].text.c_str());
    }
  }

  // Check model vocab integrity
  if (model->hparams.n_vocab != model->vocab.id_to_token.size()) {
    throw std::runtime_error("Model vocab size mismatch");
  }

  printf("\n");

#define __LOAD_MODEL_TENSORS
  size_t ctx_size = 0;
  size_t mmapped_size = 0;
  {
    for (int i = 0; i < gguf_ctx->header.n_tensors; i++) {
      struct ggml_tensor* tensor = ggml_get_tensor(ctx_meta, gguf_ctx->infos[i].name.data);
      ctx_size += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
      mmapped_size += GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
    }
    LLAMA_LOG_INFO("%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size / 1024.0 / 1024.0);

    // Create the ggml context
    {
      model->buf.resize(ctx_size);
      if (use_mlock) {
        model->mlock_buf.init(model->buf.data);
        model->mlock_buf.grow_to(model->buf.size);
      }

      struct ggml_init_params params = {
          /*.mem_size   =*/model->buf.size,
          /*.mem_buffer =*/model->buf.data,
          /*.no_alloc   =*/use_mmap,
      };

      model->ctx = ggml_init(params);
      if (!model->ctx) {
        throw std::runtime_error(format("ggml_init() failed"));
      }
    }

    // Prepare memory for the weights
    size_t vram_weights = 0;
    {
      const int64_t n_embd = model->hparams.n_embd;
      const int64_t n_embd_gqa = model->hparams.n_embd_gqa();
      const int64_t n_layer = model->hparams.n_layer;
      const int64_t n_vocab = model->hparams.n_vocab;

      // For LLM_ARCH_LLAMA only
      {
        std::string tensor_name = LLAMA_TENSOR_NAMES[LLM_TENSOR_TOKEN_EMBD] + "." + "weight";
        struct ggml_tensor* cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
        model->tok_embeddings = create_tensor_for(model->ctx, cur, GGML_BACKEND_CPU);
        n_created++;

        // output
        {
          ggml_backend backend_norm;
          ggml_backend backend_output;

          if (n_gpu_layers > int(n_layer)) {
            // norm is not performance relevant on its own but keeping it in VRAM reduces data copying
            // on Windows however this is detrimental unless everything is on the GPU
            backend_norm = LLAMA_BACKEND_OFFLOAD;
            backend_output = LLAMA_BACKEND_OFFLOAD_SPLIT;
          } else {
            backend_norm = GGML_BACKEND_CPU;
            backend_output = GGML_BACKEND_CPU;
          }

          tensor_name = LLAMA_TENSOR_NAMES[LLM_TENSOR_OUTPUT_NORM] + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          model->output_norm = create_tensor_for(model->ctx, cur, backend_norm);
          n_created++;

          tensor_name = LLAMA_TENSOR_NAMES[LLM_TENSOR_OUTPUT] + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          model->output = create_tensor_for(model->ctx, cur, backend_output);
          n_created++;

          if (backend_norm == GGML_BACKEND_GPU) {
            vram_weights += ggml_nbytes(model->output_norm);
          }
          if (backend_output == GGML_BACKEND_GPU_SPLIT) {
            vram_weights += ggml_nbytes(model->output);
          }
        }

        const uint32_t n_ff = model->hparams.n_ff;
        const int i_gpu_start = n_layer - n_gpu_layers;
        model->layers.resize(n_layer);

        for (uint32_t i = 0; i < n_layer; ++i) {
          const ggml_backend backend = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD;
          const ggml_backend backend_split = int(i) < i_gpu_start ? GGML_BACKEND_CPU : LLAMA_BACKEND_OFFLOAD_SPLIT;

          auto& layer = model->layers[i];

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_ATTN_NORM].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.attn_norm = create_tensor_for(model->ctx, cur, backend);
          n_created++;

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_ATTN_Q].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.wq = create_tensor_for(model->ctx, cur, backend_split);
          n_created++;

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_ATTN_K].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.wk = create_tensor_for(model->ctx, cur, backend_split);
          n_created++;

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_ATTN_V].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.wv = create_tensor_for(model->ctx, cur, backend_split);
          n_created++;

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_ATTN_OUT].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.wo = create_tensor_for(model->ctx, cur, backend_split);
          n_created++;

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_FFN_NORM].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.ffn_norm = create_tensor_for(model->ctx, cur, backend);
          n_created++;

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_FFN_GATE].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.w1 = create_tensor_for(model->ctx, cur, backend_split);
          n_created++;

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_FFN_DOWN].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.w2 = create_tensor_for(model->ctx, cur, backend_split);
          n_created++;

          tensor_name = format(LLAMA_TENSOR_NAMES[LLM_TENSOR_FFN_UP].c_str(), i) + "." + "weight";
          cur = ggml_get_tensor(ctx_meta, tensor_name.c_str());
          layer.w3 = create_tensor_for(model->ctx, cur, backend_split);
          n_created++;

          if (backend == GGML_BACKEND_GPU) {
            vram_weights += ggml_nbytes(layer.attn_norm) + ggml_nbytes(layer.wq) + ggml_nbytes(layer.wk) +
                            ggml_nbytes(layer.wv) + ggml_nbytes(layer.wo) + ggml_nbytes(layer.ffn_norm) +
                            ggml_nbytes(layer.w1) + ggml_nbytes(layer.w2) + ggml_nbytes(layer.w3);
          }
        }
      }
    }

    // Integrity check
    if (n_created != n_tensors) {
      throw std::runtime_error(
          format("%s: wrong number of tensors; expected %d, got %d", __func__, n_tensors, n_created));
    }

    // Print memory requirements
    {
      // this is the total memory required to run the inference
      size_t mem_required = ctx_size + mmapped_size - vram_weights;  // weights in VRAM not in memory

      LLAMA_LOG_INFO("%s: mem required  = %7.2f MB\n", __func__, mem_required / 1024.0 / 1024.0);

      // If cuda is used
      if (ggml_use_cuda()) {
        const int n_gpu = std::min(n_gpu_layers, int(model->hparams.n_layer));

        LLAMA_LOG_INFO("%s: offloading %d repeating layers to GPU\n", __func__, n_gpu);
        if (n_gpu_layers > (int)model->hparams.n_layer) {
          LLAMA_LOG_INFO("%s: offloading non-repeating layers to GPU\n", __func__);
        }

        const int max_backend_supported_layers = model->hparams.n_layer + 3;
        const int max_offloadable_layers = model->hparams.n_layer + 3;

        LLAMA_LOG_INFO("%s: offloaded %d/%d layers to GPU\n", __func__, std::min(n_gpu_layers, max_offloadable_layers),
                       max_backend_supported_layers);
        LLAMA_LOG_INFO("%s: VRAM used: %.2f MB\n", __func__, vram_weights / 1024.0 / 1024.0);
      }
    }

    // populate `tensors_by_name`
    for (int i = 0; i < n_tensors; ++i) {
      struct ggml_tensor* cur = ggml_get_tensor(model->ctx, gguf_ctx->infos[i].name.data);
      model->tensors_by_name.emplace_back(cur->name, cur);
    }

    (void)tensor_split;
#ifdef GGML_USE_CUBLAS
    { ggml_cuda_set_tensor_split(tensor_split); }
#endif

#define __DATA_FOR_TENSORS
    {
      printf("Loading data for tensors\n");
      size_t size_data = 0;
      size_t size_lock = 0;
      size_t size_pref = 0;  // prefetch

      for (int i = 0; i < n_tensors; i++) {
        struct ggml_tensor* cur = ggml_get_tensor(model->ctx, gguf_ctx->infos[i].name.data);
        size_data += ggml_nbytes(cur);
        if (cur->backend == GGML_BACKEND_CPU) {
          size_pref += ggml_nbytes(cur);
        }
      }

      std::unique_ptr<llama_mmap> mapping;
      llama_mlock* lmlock = use_mlock ? &model->mlock_mmap : NULL;
      llama_file file(model_file_path, "rb");
      if (use_mmap) {
        mapping.reset(new llama_mmap(&file, size_pref, ggml_is_numa()));
        if (lmlock) {
          lmlock->init(mapping->addr);
        }
      }

      size_t done_size = 0;
      for (int i = 0; i < n_tensors; i++) {
        struct ggml_tensor* cur = ggml_get_tensor(model->ctx, gguf_ctx->infos[i].name.data);
        GGML_ASSERT(cur);  // unused tensors should have been caught by load_data already

        unsigned cur_percentage = 0;
        llama_progress_callback progress_callback = NULL;
        void* progress_callback_user_data = NULL;
        if (progress_callback == NULL) {
          progress_callback_user_data = &cur_percentage;
          progress_callback = [](float progress, void* ctx) {
            unsigned* cur_percentage_p = (unsigned*)ctx;
            unsigned percentage = (unsigned)(100 * progress);
            while (percentage > *cur_percentage_p) {
              *cur_percentage_p = percentage;
              LLAMA_LOG_INFO(".");
              if (percentage >= 100) {
                LLAMA_LOG_INFO("\n");
              }
            }
          };
        }

        progress_callback((float)done_size / size_data, progress_callback_user_data);

        // Load data for cur
        const size_t offs = gguf_ctx->offset + gguf_ctx->infos[i].offset;
        cur->data = (uint8_t*)mapping->addr + offs;

        switch (cur->backend) {
          case GGML_BACKEND_CPU:
            if (use_mmap && lmlock) {
              size_lock += ggml_nbytes(cur);
              lmlock->grow_to(size_lock);
            }
            break;
#ifdef GGML_USE_CUBLAS
          case GGML_BACKEND_GPU:
          case GGML_BACKEND_GPU_SPLIT:
            // old code:
            // ggml_cuda_transform_tensor(lt.data, lt.ggml_tensor);

            // TODO: test if this works !!
            ggml_cuda_transform_tensor(cur->data, cur);
            if (!use_mmap) {
              free(cur->data);
            }
            break;
#endif
          default:
            continue;
        }

        done_size += ggml_nbytes(cur);
      }
    }
  }

  printf("\n");
}