#include <pthread.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>

#include <atomic>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <fstream>
#include <map>
#include <memory>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "fundamentals.h"
#include "ggml-cuda.h"
#include "llama.h"

#define __GLOBAL_VARIABLES__
// =======================================================================
// Global state
// =======================================================================
static struct ggml_state g_state;
// static atomic_int g_state_barrier = 0;

#define __UTILITY_FUNCTIONS__
// =======================================================================
// Utility functions
// =======================================================================
static bool ggml_use_cublas() { return std::getenv("GGML_USE_CUBLAS") ? true : false; }

size_t ggml_used_mem(const struct ggml_context* ctx) {
  return ctx->objects_end == NULL ? 0 : ctx->objects_end->offs + ctx->objects_end->size;
}

bool ggml_is_numa(void) { return g_state.numa.n_nodes > 1; }

static void set_numa_thread_affinity(int thread_n, int n_threads) {
  if (!ggml_is_numa()) {
    return;
  }

  // run thread on node_num thread_n / (threads per node)
  const int node_num = thread_n / ((n_threads + g_state.numa.n_nodes - 1) / g_state.numa.n_nodes);
  struct ggml_numa_node* node = &g_state.numa.nodes[node_num];
  size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

  cpu_set_t* cpus = CPU_ALLOC(g_state.numa.total_cpus);
  CPU_ZERO_S(setsize, cpus);
  for (size_t i = 0; i < node->n_cpus; ++i) {
    CPU_SET_S(node->cpus[i], setsize, cpus);
  }

  int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
  if (rv) {
    fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
  }

  CPU_FREE(cpus);
}

static void clear_numa_thread_affinity(void) {
  if (!ggml_is_numa()) {
    return;
  }

  size_t setsize = CPU_ALLOC_SIZE(g_state.numa.total_cpus);

  cpu_set_t* cpus = CPU_ALLOC(g_state.numa.total_cpus);
  CPU_ZERO_S(setsize, cpus);
  for (unsigned i = 0; i < g_state.numa.total_cpus; ++i) {
    CPU_SET_S(i, setsize, cpus);
  }

  int rv = pthread_setaffinity_np(pthread_self(), setsize, cpus);
  if (rv) {
    fprintf(stderr, "warning: pthread_setaffinity_np() failed: %s\n", strerror(rv));
  }

  CPU_FREE(cpus);
}

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

#ifdef __GNUC__
#define LLAMA_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
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

static size_t aligned_offset(const void* buffer, size_t offset, size_t alignment) {
  assert(alignment && !(alignment & (alignment - 1)));  // power of 2
  size_t align = (alignment - (((uintptr_t)buffer + offset) % alignment)) % alignment;
  return offset + align;
}

std::string build_llm_key(const std::string& original_key, const std::string& arch_name) {
  if (original_key.find("%s") >= 0) {
    char* key = (char*)calloc((original_key.length() - 2) + arch_name.length() + 1, 1);
    sprintf(key, original_key.c_str(), arch_name.c_str());
    return key;
  }

  return original_key;
}

static size_t hash(void* p) { return (size_t)p % GGML_GRAPH_HASHTABLE_SIZE; }

static struct hash_node* hash_get(struct hash_node hash_table[], struct ggml_tensor* t) {
  size_t h = hash(t);

  // linear probing
  size_t i = h;
  while (hash_table[i].t != NULL) {
    if (hash_table[i].t == t) {
      return &hash_table[i];
    }
    i = (i + 1) % GGML_GRAPH_HASHTABLE_SIZE;
    if (i == h) {
      // hash table is full
      GGML_ASSERT(false);
    }
  }

  hash_table[i].t = t;
  return &hash_table[i];
}

static size_t hash_find(void* hash_table[], void* p) {
  size_t h = hash(p);

  // linear probing
  size_t i = h;
  while (hash_table[i] != NULL && hash_table[i] != p) {
    i = (i + 1) % GGML_GRAPH_HASHTABLE_SIZE;
    if (i == h) {
      // visited all hash table entries -> not found
      return GGML_GRAPH_HASHTABLE_SIZE;
    }
  }
  return i;
}

static bool hash_insert(void* hash_table[], void* p) {
  size_t i = hash_find(hash_table, p);

  GGML_ASSERT(i < GGML_GRAPH_HASHTABLE_SIZE);  // assert that not full

  if (hash_table[i] == p) {
    return true;
  }

  // insert
  GGML_ASSERT(hash_table[i] == NULL);
  hash_table[i] = p;
  return false;
}

#define __GGML_COMMON_FUNCTIONS__
// =======================================================================
// GGML functions
// =======================================================================
void ggml_numa_init(void) {
  if (g_state.numa.n_nodes > 0) {
    fprintf(stderr, "ggml_numa_init: NUMA already initialized\n");
    return;
  }

  struct stat st;
  char path[256];
  int rv;

  // Enumerate nodes
  while (g_state.numa.n_nodes < GGML_NUMA_MAX_NODES) {
    rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u", g_state.numa.n_nodes);
    GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
    if (stat(path, &st) != 0) {
      break;
    }
    ++g_state.numa.n_nodes;
  }

  // Enumerate CPUs
  while (g_state.numa.total_cpus < GGML_NUMA_MAX_CPUS) {
    rv = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u", g_state.numa.total_cpus);
    GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
    if (stat(path, &st) != 0) {
      break;
    }
    ++g_state.numa.total_cpus;
  }

  GGML_PRINT_DEBUG("found %u numa nodes, %u CPUs\n", g_state.numa.n_nodes, g_state.numa.total_cpus);

  if (g_state.numa.n_nodes < 1 || g_state.numa.total_cpus < 1) {
    g_state.numa.n_nodes = 0;
    return;
  }

  for (uint32_t n = 0; n < g_state.numa.n_nodes; ++n) {
    struct ggml_numa_node* node = &g_state.numa.nodes[n];
    GGML_PRINT_DEBUG("CPUs on node %u:", n);
    node->n_cpus = 0;
    for (uint32_t c = 0; c < g_state.numa.total_cpus; ++c) {
      rv = snprintf(path, sizeof(path), "/sys/devices/system/node/node%u/cpu%u", n, c);
      GGML_ASSERT(rv > 0 && (unsigned)rv < sizeof(path));
      if (stat(path, &st) == 0) {
        node->cpus[node->n_cpus++] = c;
        GGML_PRINT_DEBUG(" %u", c);
      }
    }
    GGML_PRINT_DEBUG("\n");
  }

  if (ggml_is_numa()) {
    FILE* fptr = fopen("/proc/sys/kernel/numa_balancing", "r");
    if (fptr != NULL) {
      char buf[42];
      if (fgets(buf, sizeof(buf), fptr) && strncmp(buf, "0\n", sizeof(buf)) != 0) {
        GGML_PRINT(
            "WARNING: /proc/sys/kernel/numa_balancing is enabled, this has been observed to impair performance\n");
      }
      fclose(fptr);
    }
  }
}

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

    if (ggml_use_cublas()) {
      ggml_init_cublas();
    }

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

static bool ggml_show_perf() { return std::getenv("GGML_PERF") ? true : false; }

static inline bool ggml_is_vector(const struct ggml_tensor* tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ggml_is_matrix(const struct ggml_tensor* tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

struct ggml_tensor* ggml_get_rows(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
  GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);

  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  // TODO: implement non F32 return
  // struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
  struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, new int64_t[2]{a->ne[0], b->ne[0]});

  result->op = GGML_OP_GET_ROWS;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor* ggml_set_f32(struct ggml_tensor* tensor, float value) {
  const int n = tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
  const int nc = tensor->ne[0];
  const size_t n1 = tensor->nb[1];

  char* const data = (char*)tensor->data;

  switch (tensor->type) {
    case GGML_TYPE_I8: {
      assert(tensor->nb[0] == sizeof(int8_t));
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i8(nc, (int8_t*)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_I16: {
      assert(tensor->nb[0] == sizeof(int16_t));
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i16(nc, (int16_t*)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_I32: {
      assert(tensor->nb[0] == sizeof(int32_t));
      for (int i = 0; i < n; i++) {
        ggml_vec_set_i32(nc, (int32_t*)(data + i * n1), value);
      }
    } break;
    case GGML_TYPE_F16: {
      assert(tensor->nb[0] == sizeof(ggml_fp16_t));
      for (int i = 0; i < n; i++) {
        ggml_vec_set_f16(nc, (ggml_fp16_t*)(data + i * n1), GGML_FP32_TO_FP16(value));
      }
    } break;
    case GGML_TYPE_F32: {
      assert(tensor->nb[0] == sizeof(float));
      for (int i = 0; i < n; i++) {
        ggml_vec_set_f32(nc, (float*)(data + i * n1), value);
      }
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }

  return tensor;
}

struct ggml_tensor* ggml_format_name(struct ggml_tensor* tensor, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(tensor->name, sizeof(tensor->name), fmt, args);
  va_end(args);
  return tensor;
}

struct ggml_tensor* ggml_view_tensor(struct ggml_context* ctx, struct ggml_tensor* src) {
  struct ggml_tensor* result = ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src, 0);
  ggml_format_name(result, "%s (view)", src->name);

  for (int i = 0; i < GGML_MAX_DIMS; i++) {
    result->nb[i] = src->nb[i];
  }

  return result;
}

static struct ggml_tensor* ggml_rope_impl(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b,
                                          int n_dims, int mode, int n_ctx, float freq_base, float freq_scale,
                                          float xpos_base, bool xpos_down, bool inplace) {
  GGML_ASSERT(ggml_is_vector(b));
  GGML_ASSERT(b->type == GGML_TYPE_I32);
  GGML_ASSERT(a->ne[2] == b->ne[0]);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

  int32_t params[8] = {/*n_past*/ 0, n_dims, mode, n_ctx};
  memcpy(params + 4, &freq_base, sizeof(float));
  memcpy(params + 5, &freq_scale, sizeof(float));
  memcpy(params + 6, &xpos_base, sizeof(float));
  memcpy(params + 7, &xpos_down, sizeof(bool));

  GGML_ASSERT(result != NULL);  // silence -Warray-bounds warnings
  assert(sizeof(params) <= GGML_MAX_OP_PARAMS);
  memcpy(result->op_params, params, sizeof(params));

  result->op = GGML_OP_ROPE;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor* ggml_rope_custom_inplace(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b,
                                             int n_dims, int mode, int n_ctx, float freq_base, float freq_scale) {
  return ggml_rope_impl(ctx, a, b, n_dims, mode, n_ctx, freq_base, freq_scale, 0.0f, false, true);
}

struct ggml_tensor* ggml_rope_custom(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b, int n_dims,
                                     int mode, int n_ctx, float freq_base, float freq_scale) {
  return ggml_rope_impl(ctx, a, b, n_dims, mode, n_ctx, freq_base, freq_scale, 0.0f, false, false);
}

static struct ggml_tensor* ggml_view_impl(struct ggml_context* ctx, struct ggml_tensor* a, int n_dims,
                                          const int64_t* ne, size_t offset) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor* result = ggml_new_tensor_impl(ctx, a->type, n_dims, ne, a, offset);
  ggml_format_name(result, "%s (view)", a->name);

  GGML_ASSERT(result != NULL);  // silence -Warray-bounds warnings
  assert(sizeof(offset) <= GGML_MAX_OP_PARAMS);
  memcpy(result->op_params, &offset, sizeof(offset));

  result->op = GGML_OP_VIEW;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

// ggml_view_1d
struct ggml_tensor* ggml_view_1d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0, size_t offset) {
  struct ggml_tensor* result = ggml_view_impl(ctx, a, 1, &ne0, offset);

  return result;
}

// ggml_view_2d
struct ggml_tensor* ggml_view_2d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0, int64_t ne1, size_t nb1,
                                 size_t offset) {
  const int64_t ne[2] = {ne0, ne1};

  struct ggml_tensor* result = ggml_view_impl(ctx, a, 2, ne, offset);

  result->nb[1] = nb1;
  result->nb[2] = result->nb[1] * ne1;
  result->nb[3] = result->nb[2];

  return result;
}

// ggml_view_3d
struct ggml_tensor* ggml_view_3d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2,
                                 size_t nb1, size_t nb2, size_t offset) {
  const int64_t ne[3] = {ne0, ne1, ne2};

  struct ggml_tensor* result = ggml_view_impl(ctx, a, 3, ne, offset);

  result->nb[1] = nb1;
  result->nb[2] = nb2;
  result->nb[3] = result->nb[2] * ne2;

  return result;
}

static void ggml_visit_parents(struct ggml_cgraph* cgraph, struct ggml_tensor* node) {
  if (node->grad == NULL) {
    // this usually happens when we generate intermediate nodes from constants in the backward pass
    // it can also happen during forward pass, if the user performs computations with constants
    if (node->op != GGML_OP_NONE) {
      // GGML_PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
    }
  }

  // check if already visited
  if (hash_insert(cgraph->visited_hash_table, node)) {
    return;
  }

  for (int i = 0; i < GGML_MAX_SRC; ++i) {
    const int k = (cgraph->order == GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT) ? i
                  : (cgraph->order == GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT)
                      ? (GGML_MAX_SRC - 1 - i)
                      :
                      /* unknown order, just fall back to using i*/ i;
    if (node->src[k]) {
      ggml_visit_parents(cgraph, node->src[k]);
    }
  }

  if (node->op == GGML_OP_NONE && node->grad == NULL) {
    // reached a leaf node, not part of the gradient graph (e.g. a constant)
    GGML_ASSERT(cgraph->n_leafs < GGML_MAX_NODES);

    if (strlen(node->name) == 0) {
      ggml_format_name(node, "leaf_%d", cgraph->n_leafs);
    }

    cgraph->leafs[cgraph->n_leafs] = node;
    cgraph->n_leafs++;
  } else {
    GGML_ASSERT(cgraph->n_nodes < GGML_MAX_NODES);

    if (strlen(node->name) == 0) {
      ggml_format_name(node, "node_%d", cgraph->n_nodes);
    }

    cgraph->nodes[cgraph->n_nodes] = node;
    cgraph->grads[cgraph->n_nodes] = node->grad;
    cgraph->n_nodes++;
  }
}

static void ggml_build_forward_impl(struct ggml_cgraph* cgraph, struct ggml_tensor* tensor, bool expand) {
  if (!expand) {
    cgraph->n_nodes = 0;
    cgraph->n_leafs = 0;
  }

  const int n0 = cgraph->n_nodes;
  // UNUSED(n0);

  ggml_visit_parents(cgraph, tensor);

  const int n_new = cgraph->n_nodes - n0;
  GGML_PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

  if (n_new > 0) {
    // the last added node should always be starting point
    GGML_ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
  }
}

static struct ggml_tensor* ggml_rms_norm_impl(struct ggml_context* ctx, struct ggml_tensor* a, float eps,
                                              bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

  GGML_ASSERT(result != NULL);  // silence -Warray-bounds warnings
  assert(sizeof(eps) <= GGML_MAX_OP_PARAMS);
  memcpy(result->op_params, &eps, sizeof(eps));

  result->op = GGML_OP_RMS_NORM;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

struct ggml_tensor* ggml_rms_norm(struct ggml_context* ctx, struct ggml_tensor* a, float eps) {
  return ggml_rms_norm_impl(ctx, a, eps, false);
}

// check if t1 can be represented as a repeatition of t0
static inline bool ggml_can_repeat(const struct ggml_tensor* t0, const struct ggml_tensor* t1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (t1->ne[0] % t0->ne[0] == 0) && (t1->ne[1] % t0->ne[1] == 0) && (t1->ne[2] % t0->ne[2] == 0) &&
         (t1->ne[3] % t0->ne[3] == 0);
}

static inline bool ggml_can_repeat_rows(const struct ggml_tensor* t0, const struct ggml_tensor* t1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (t0->ne[0] == t1->ne[0]) && ggml_can_repeat(t0, t1);
}

bool ggml_are_same_shape(const struct ggml_tensor* t0, const struct ggml_tensor* t1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (t0->ne[0] == t1->ne[0]) && (t0->ne[1] == t1->ne[1]) && (t0->ne[2] == t1->ne[2]) && (t0->ne[3] == t1->ne[3]);
}

static struct ggml_tensor* ggml_mul_impl(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b,
                                         bool inplace) {
  // TODO: support less-strict constraint
  //       GGML_ASSERT(ggml_can_repeat(b, a));
  GGML_ASSERT(ggml_can_repeat_rows(b, a));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    // TODO: support backward pass for broadcasting
    GGML_ASSERT(ggml_are_same_shape(a, b));
    is_node = true;
  }

  if (inplace) {
    GGML_ASSERT(!is_node);
  }

  struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

  result->op = GGML_OP_MUL;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

static inline bool ggml_can_mul_mat(const struct ggml_tensor* t0, const struct ggml_tensor* t1) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return (t0->ne[0] == t1->ne[0]) && (t1->ne[2] % t0->ne[2] == 0) &&  // verify t0 is broadcastable
         (t1->ne[3] % t0->ne[3] == 0);
}

struct ggml_tensor* ggml_mul(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
  return ggml_mul_impl(ctx, a, b, false);
}

bool ggml_is_transposed(const struct ggml_tensor* tensor) { return tensor->nb[0] > tensor->nb[1]; }

struct ggml_tensor* ggml_mul_mat(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
  GGML_ASSERT(ggml_can_mul_mat(a, b));
  GGML_ASSERT(!ggml_is_transposed(a));

  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  const int64_t ne[4] = {a->ne[1], b->ne[1], b->ne[2], b->ne[3]};
  struct ggml_tensor* result = ggml_new_tensor(ctx, GGML_TYPE_F32, MAX(a->n_dims, b->n_dims), ne);

  result->op = GGML_OP_MUL_MAT;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

bool ggml_is_contiguous(const struct ggml_tensor* tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->nb[0] == type_traits[tensor->type].type_size &&
         tensor->nb[1] == (tensor->nb[0] * tensor->ne[0]) / type_traits[tensor->type].blck_size &&
         tensor->nb[2] == tensor->nb[1] * tensor->ne[1] && tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

static inline bool ggml_is_scalar(const struct ggml_tensor* tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static inline bool ggml_is_padded_1d(const struct ggml_tensor* tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->nb[0] == type_traits[tensor->type].type_size && tensor->nb[2] == tensor->nb[1] * tensor->ne[1] &&
         tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

int64_t ggml_nelements(const struct ggml_tensor* tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

struct ggml_tensor* ggml_reshape_1d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0) {
  GGML_ASSERT(ggml_is_contiguous(a));
  GGML_ASSERT(ggml_nelements(a) == ne0);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[1] = {ne0};
  struct ggml_tensor* result = ggml_new_tensor_impl(ctx, a->type, 1, ne, a, 0);
  ggml_format_name(result, "%s (reshaped)", a->name);

  result->op = GGML_OP_RESHAPE;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

struct ggml_tensor* ggml_reshape_2d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0, int64_t ne1) {
  GGML_ASSERT(ggml_is_contiguous(a));
  GGML_ASSERT(ggml_nelements(a) == ne0 * ne1);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[2] = {ne0, ne1};
  struct ggml_tensor* result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a, 0);
  ggml_format_name(result, "%s (reshaped)", a->name);

  result->op = GGML_OP_RESHAPE;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

struct ggml_tensor* ggml_reshape_3d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0, int64_t ne1,
                                    int64_t ne2) {
  GGML_ASSERT(ggml_is_contiguous(a));
  GGML_ASSERT(a->ne[0] * a->ne[1] * a->ne[2] * a->ne[3] == ne0 * ne1 * ne2);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  const int64_t ne[3] = {ne0, ne1, ne2};
  struct ggml_tensor* result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a, 0);
  ggml_format_name(result, "%s (reshaped)", a->name);

  result->op = GGML_OP_RESHAPE;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

struct ggml_tensor* ggml_transpose(struct ggml_context* ctx, struct ggml_tensor* a) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor* result = ggml_view_tensor(ctx, a);
  ggml_format_name(result, "%s (transposed)", a->name);

  result->ne[0] = a->ne[1];
  result->ne[1] = a->ne[0];

  result->nb[0] = a->nb[1];
  result->nb[1] = a->nb[0];

  result->op = GGML_OP_TRANSPOSE;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

struct ggml_tensor* ggml_permute(struct ggml_context* ctx, struct ggml_tensor* a, int axis0, int axis1, int axis2,
                                 int axis3) {
  GGML_ASSERT(axis0 >= 0 && axis0 < GGML_MAX_DIMS);
  GGML_ASSERT(axis1 >= 0 && axis1 < GGML_MAX_DIMS);
  GGML_ASSERT(axis2 >= 0 && axis2 < GGML_MAX_DIMS);
  GGML_ASSERT(axis3 >= 0 && axis3 < GGML_MAX_DIMS);

  GGML_ASSERT(axis0 != axis1);
  GGML_ASSERT(axis0 != axis2);
  GGML_ASSERT(axis0 != axis3);
  GGML_ASSERT(axis1 != axis2);
  GGML_ASSERT(axis1 != axis3);
  GGML_ASSERT(axis2 != axis3);

  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor* result = ggml_view_tensor(ctx, a);
  ggml_format_name(result, "%s (permuted)", a->name);

  int ne[GGML_MAX_DIMS];
  int nb[GGML_MAX_DIMS];

  ne[axis0] = a->ne[0];
  ne[axis1] = a->ne[1];
  ne[axis2] = a->ne[2];
  ne[axis3] = a->ne[3];

  nb[axis0] = a->nb[0];
  nb[axis1] = a->nb[1];
  nb[axis2] = a->nb[2];
  nb[axis3] = a->nb[3];

  result->ne[0] = ne[0];
  result->ne[1] = ne[1];
  result->ne[2] = ne[2];
  result->ne[3] = ne[3];

  result->nb[0] = nb[0];
  result->nb[1] = nb[1];
  result->nb[2] = nb[2];
  result->nb[3] = nb[3];

  result->op = GGML_OP_PERMUTE;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  int32_t params[] = {axis0, axis1, axis2, axis3};
  GGML_ASSERT(result != NULL);  // silence -Warray-bounds warnings
  assert(sizeof(params) <= GGML_MAX_OP_PARAMS);
  memcpy(result->op_params, params, sizeof(params));

  return result;
}

static struct ggml_tensor* ggml_cpy_impl(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b,
                                         bool inplace) {
  GGML_ASSERT(ggml_nelements(a) == ggml_nelements(b));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    is_node = true;
  }

  // make a view of the destination
  struct ggml_tensor* result = ggml_view_tensor(ctx, b);
  if (strlen(b->name) > 0) {
    ggml_format_name(result, "%s (copy of %s)", b->name, a->name);
  } else {
    ggml_format_name(result, "%s (copy)", a->name);
  }

  result->op = GGML_OP_CPY;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor* ggml_cpy(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
  return ggml_cpy_impl(ctx, a, b, false);
}

static struct ggml_tensor* ggml_scale_impl(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b,
                                           bool inplace) {
  GGML_ASSERT(ggml_is_scalar(b));
  GGML_ASSERT(ggml_is_padded_1d(a));

  bool is_node = false;

  if (a->grad || b->grad) {
    is_node = true;
  }

  struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

  result->op = GGML_OP_SCALE;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor* ggml_scale(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
  return ggml_scale_impl(ctx, a, b, false);
}

static struct ggml_tensor* ggml_add_impl(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b,
                                         bool inplace) {
  // TODO: support less-strict constraint
  //       GGML_ASSERT(ggml_can_repeat(b, a));
  GGML_ASSERT(ggml_can_repeat_rows(b, a));

  bool is_node = false;

  if (!inplace && (a->grad || b->grad)) {
    // TODO: support backward pass for broadcasting
    GGML_ASSERT(ggml_are_same_shape(a, b));
    is_node = true;
  }

  struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

  result->op = GGML_OP_ADD;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;
  result->src[1] = b;

  return result;
}

struct ggml_tensor* ggml_add(struct ggml_context* ctx, struct ggml_tensor* a, struct ggml_tensor* b) {
  return ggml_add_impl(ctx, a, b, false);
}

static struct ggml_tensor* ggml_soft_max_impl(struct ggml_context* ctx, struct ggml_tensor* a, bool inplace) {
  bool is_node = false;

  if (a->grad) {
    is_node = true;
  }

  struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);

  result->op = GGML_OP_SOFT_MAX;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

struct ggml_tensor* ggml_soft_max(struct ggml_context* ctx, struct ggml_tensor* a) {
  return ggml_soft_max_impl(ctx, a, false);
}

// make contiguous, with new shape
struct ggml_tensor* ggml_cont_4d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0, int64_t ne1, int64_t ne2,
                                 int64_t ne3) {
  GGML_ASSERT(ggml_nelements(a) == (ne0 * ne1 * ne2 * ne3));

  bool is_node = false;

  struct ggml_tensor* result = ggml_new_tensor(ctx, a->type, 4, new int64_t[4]{ne0, ne1, ne2, ne3});
  ggml_format_name(result, "%s (cont)", a->name);

  result->op = GGML_OP_CONT;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

GGML_API struct ggml_tensor* ggml_cont_1d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0) {
  return ggml_cont_4d(ctx, a, ne0, 1, 1, 1);
}

GGML_API struct ggml_tensor* ggml_cont_2d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0, int64_t ne1) {
  return ggml_cont_4d(ctx, a, ne0, ne1, 1, 1);
}

GGML_API struct ggml_tensor* ggml_cont_3d(struct ggml_context* ctx, struct ggml_tensor* a, int64_t ne0, int64_t ne1,
                                          int64_t ne2) {
  return ggml_cont_4d(ctx, a, ne0, ne1, ne2, 1);
}

static struct ggml_tensor* ggml_unary_impl(struct ggml_context* ctx, struct ggml_tensor* a, enum ggml_unary_op op,
                                           bool inplace) {
  bool is_node = false;

  if (!inplace && (a->grad)) {
    is_node = true;
  }

  struct ggml_tensor* result = inplace ? ggml_view_tensor(ctx, a) : ggml_new_tensor(ctx, a->type, a->n_dims, a->ne);
  assert(0 < GGML_MAX_OP_PARAMS / sizeof(int32_t));
  ((int32_t*)(result->op_params))[0] = (int32_t)op;

  result->op = GGML_OP_UNARY;
  result->grad = is_node ? ggml_new_tensor(ctx, result->type, result->n_dims, result->ne) : NULL;
  result->src[0] = a;

  return result;
}

struct ggml_tensor* ggml_unary(struct ggml_context* ctx, struct ggml_tensor* a, enum ggml_unary_op op) {
  return ggml_unary_impl(ctx, a, op, false);
}

struct ggml_tensor* ggml_silu(struct ggml_context* ctx, struct ggml_tensor* a) {
  return ggml_unary(ctx, a, GGML_UNARY_OP_SILU);
}

static bool ggml_op_can_inplace(enum ggml_op op) {
  switch (op) {
    case GGML_OP_SCALE:
    case GGML_OP_DIAG_MASK_ZERO:
    case GGML_OP_DIAG_MASK_INF:
    case GGML_OP_ADD:
    case GGML_OP_ADD1:
    case GGML_OP_SUB:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
    case GGML_OP_SQR:
    case GGML_OP_SQRT:
    case GGML_OP_LOG:
    case GGML_OP_UNARY:
    case GGML_OP_ROPE:
    case GGML_OP_RMS_NORM:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_CONT:
      return true;

    default:
      return false;
  }
}

static bool ggml_are_same_layout(const struct ggml_tensor* a, const struct ggml_tensor* b) {
  if (a->type != b->type) {
    return false;
  }
  for (int i = 0; i < GGML_MAX_DIMS; i++) {
    if (a->ne[i] != b->ne[i]) {
      return false;
    }
    if (a->nb[i] != b->nb[i]) {
      return false;
    }
  }
  return true;
}

void ggml_time_init(void) {}
int64_t ggml_time_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + (int64_t)ts.tv_nsec / 1000000;
}

int64_t ggml_time_us(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000 + (int64_t)ts.tv_nsec / 1000;
}

int64_t ggml_cycles(void) { return clock(); }

int64_t ggml_cycles_per_ms(void) { return CLOCKS_PER_SEC / 1000; }

void ggml_graph_print(const struct ggml_cgraph* cgraph) {
  int64_t perf_total_per_op_us[GGML_OP_COUNT] = {0};

  GGML_PRINT("=== GRAPH ===\n");

  GGML_PRINT("n_nodes = %d\n", cgraph->n_nodes);
  for (int i = 0; i < cgraph->n_nodes; i++) {
    struct ggml_tensor* node = cgraph->nodes[i];

    perf_total_per_op_us[node->op] += MAX(1, node->perf_time_us);

    GGML_PRINT(" - %3d: [ %5" PRId64 ", %5" PRId64 ", %5" PRId64
               "] %16s %s (%3d) cpu = %7.3f / %7.3f ms, wall = %7.3f / %7.3f ms\n",
               i, node->ne[0], node->ne[1], node->ne[2], GGML_OP_NAMES[node->op],
               node->is_param ? "x"
               : node->grad   ? "g"
                              : " ",
               node->perf_runs, (double)node->perf_cycles / (double)ggml_cycles_per_ms(),
               (double)node->perf_cycles / (double)ggml_cycles_per_ms() / (double)node->perf_runs,
               (double)node->perf_time_us / 1000.0, (double)node->perf_time_us / 1000.0 / node->perf_runs);
  }

  GGML_PRINT("n_leafs = %d\n", cgraph->n_leafs);
  for (int i = 0; i < cgraph->n_leafs; i++) {
    struct ggml_tensor* node = cgraph->leafs[i];

    GGML_PRINT(" - %3d: [ %5" PRId64 ", %5" PRId64 "] %8s %16s\n", i, node->ne[0], node->ne[1], GGML_OP_NAMES[node->op],
               node->name);
  }

  for (int i = 0; i < GGML_OP_COUNT; i++) {
    if (perf_total_per_op_us[i] == 0) {
      continue;
    }

    GGML_PRINT("perf_total_per_op_us[%16s] = %7.3f ms\n", GGML_OP_NAMES[i], (double)perf_total_per_op_us[i] / 1000.0);
  }

  GGML_PRINT("========================================\n");
}

enum ggml_unary_op ggml_get_unary_op(const struct ggml_tensor* tensor) {
  GGML_ASSERT(tensor->op == GGML_OP_UNARY);
  assert(0 < GGML_MAX_OP_PARAMS / sizeof(int32_t));
  return (enum ggml_unary_op)((const int32_t*)(tensor->op_params))[0];
}

static inline int ggml_up32(int n) { return (n + 31) & ~31; }

static inline int ggml_up(int n, int m) {
  // assert m is a power of 2
  GGML_ASSERT((m & (m - 1)) == 0);
  return (n + m - 1) & ~(m - 1);
}

int64_t ggml_nrows(const struct ggml_tensor* tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

static int32_t ggml_get_op_params_i32(const struct ggml_tensor* tensor, uint32_t i) {
  assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
  return ((const int32_t*)(tensor->op_params))[i];
}

static inline bool ggml_is_contiguous_except_dim_1(const struct ggml_tensor* tensor) {
  static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

  return tensor->nb[0] == type_traits[tensor->type].type_size && tensor->nb[2] == tensor->nb[1] * tensor->ne[1] &&
         tensor->nb[3] == tensor->nb[2] * tensor->ne[2];
}

size_t ggml_tensor_overhead() { return GGML_OBJECT_SIZE + GGML_TENSOR_SIZE; }
size_t ggml_graph_overhead() { return GGML_OBJECT_SIZE + GGML_PAD(GGML_GRAPH_SIZE, GGML_MEM_ALIGN); }

struct ggml_cplan ggml_graph_plan(struct ggml_cgraph* cgraph, int n_threads) {
  if (n_threads <= 0) {
    n_threads = GGML_DEFAULT_N_THREADS;
  }

  size_t work_size = 0;

  struct ggml_cplan cplan;
  memset(&cplan, 0, sizeof(struct ggml_cplan));

  // thread scheduling for the different operations + work buffer size estimation
  for (int i = 0; i < cgraph->n_nodes; i++) {
    int n_tasks = 1;

    struct ggml_tensor* node = cgraph->nodes[i];

    switch (node->op) {
      case GGML_OP_CPY:
      case GGML_OP_DUP: {
        n_tasks = n_threads;

        size_t cur = 0;
        if (type_traits[node->type].is_quantized) {
          cur = type_traits[GGML_TYPE_F32].type_size * node->ne[0] * n_tasks;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_ADD:
      case GGML_OP_ADD1: {
        n_tasks = n_threads;

        size_t cur = 0;

        if (type_traits[node->src[0]->type].is_quantized) {
          cur = type_traits[GGML_TYPE_F32].type_size * node->src[0]->ne[0] * n_tasks;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_ACC: {
        n_tasks = n_threads;

        size_t cur = 0;

        if (type_traits[node->src[0]->type].is_quantized) {
          cur = type_traits[GGML_TYPE_F32].type_size * node->src[1]->ne[0] * n_tasks;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_SUB:
      case GGML_OP_DIV:
      case GGML_OP_SQR:
      case GGML_OP_SQRT:
      case GGML_OP_LOG:
      case GGML_OP_SUM:
      case GGML_OP_SUM_ROWS:
      case GGML_OP_MEAN:
      case GGML_OP_ARGMAX:
      case GGML_OP_REPEAT:
      case GGML_OP_REPEAT_BACK: {
        n_tasks = 1;
      } break;

      case GGML_OP_UNARY: {
        switch (ggml_get_unary_op(node)) {
          case GGML_UNARY_OP_ABS:
          case GGML_UNARY_OP_SGN:
          case GGML_UNARY_OP_NEG:
          case GGML_UNARY_OP_STEP:
          case GGML_UNARY_OP_TANH:
          case GGML_UNARY_OP_ELU:
          case GGML_UNARY_OP_RELU: {
            n_tasks = 1;
          } break;

          case GGML_UNARY_OP_GELU:
          case GGML_UNARY_OP_GELU_QUICK:
          case GGML_UNARY_OP_SILU: {
            n_tasks = n_threads;
          } break;
        }
      } break;
      case GGML_OP_SILU_BACK:
      case GGML_OP_MUL:
      case GGML_OP_NORM:
      case GGML_OP_RMS_NORM:
      case GGML_OP_RMS_NORM_BACK:
      case GGML_OP_GROUP_NORM: {
        n_tasks = n_threads;
      } break;
      case GGML_OP_CONCAT:
      case GGML_OP_MUL_MAT: {
        n_tasks = n_threads;

        // TODO: use different scheduling for different matrix sizes
        // const int nr0 = ggml_nrows(node->src[0]);
        // const int nr1 = ggml_nrows(node->src[1]);

        // n_tasks = MIN(n_threads, MAX(1, nr0/128));
        // printf("nr0 = %8d, nr1 = %8d, nr0*nr1 = %8d, n_tasks%d\n", nr0, nr1, nr0*nr1, n_tasks);

        size_t cur = 0;
        const enum ggml_type vec_dot_type = type_traits[node->src[0]->type].vec_dot_type;

        if (ggml_use_cublas() && ggml_cuda_can_mul_mat(node->src[0], node->src[1], node)) {
          n_tasks = 1;  // TODO: this actually is doing nothing
                        //       the threads are still spinning
        } else if (node->src[1]->type != vec_dot_type) {
          cur =
              type_traits[vec_dot_type].type_size * ggml_nelements(node->src[1]) / type_traits[vec_dot_type].blck_size;
        } else {
          cur = 0;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_OUT_PROD: {
        n_tasks = n_threads;

        size_t cur = 0;

        if (type_traits[node->src[0]->type].is_quantized) {
          cur = type_traits[GGML_TYPE_F32].type_size * node->src[0]->ne[0] * n_tasks;
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_SCALE: {
        n_tasks = 1;
      } break;
      case GGML_OP_SET:
      case GGML_OP_CONT:
      case GGML_OP_RESHAPE:
      case GGML_OP_VIEW:
      case GGML_OP_PERMUTE:
      case GGML_OP_TRANSPOSE:
      case GGML_OP_GET_ROWS:
      case GGML_OP_GET_ROWS_BACK:
      case GGML_OP_DIAG: {
        n_tasks = 1;
      } break;
      case GGML_OP_DIAG_MASK_ZERO:
      case GGML_OP_DIAG_MASK_INF:
      case GGML_OP_SOFT_MAX:
      case GGML_OP_SOFT_MAX_BACK:
      case GGML_OP_ROPE:
      case GGML_OP_ROPE_BACK:
      case GGML_OP_ADD_REL_POS: {
        n_tasks = n_threads;
      } break;
      case GGML_OP_ALIBI: {
        n_tasks = 1;  // TODO
      } break;
      case GGML_OP_CLAMP: {
        n_tasks = 1;  // TODO
      } break;
      case GGML_OP_CONV_1D: {
        n_tasks = n_threads;

        GGML_ASSERT(node->src[0]->ne[3] == 1);
        GGML_ASSERT(node->src[1]->ne[2] == 1);
        GGML_ASSERT(node->src[1]->ne[3] == 1);

        size_t cur = 0;
        const int nk = node->src[0]->ne[0];

        if (node->src[0]->type == GGML_TYPE_F16 && node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(ggml_fp16_t) * (nk * ggml_up32(node->src[0]->ne[1]) * node->src[0]->ne[2] +
                                       (2 * (nk / 2) + node->src[1]->ne[0]) * node->src[1]->ne[1]);
        } else if (node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * (nk * ggml_up32(node->src[0]->ne[1]) * node->src[0]->ne[2] +
                                 (2 * (nk / 2) + node->src[1]->ne[0]) * node->src[1]->ne[1]);
        } else {
          GGML_ASSERT(false);
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_CONV_2D: {
        n_tasks = n_threads;

        const int64_t ne00 = node->src[0]->ne[0];  // W
        const int64_t ne01 = node->src[0]->ne[1];  // H
        const int64_t ne02 = node->src[0]->ne[2];  // C
        const int64_t ne03 = node->src[0]->ne[3];  // N

        const int64_t ne10 = node->src[1]->ne[0];  // W
        const int64_t ne11 = node->src[1]->ne[1];  // H
        const int64_t ne12 = node->src[1]->ne[2];  // C

        const int64_t ne0 = node->ne[0];
        const int64_t ne1 = node->ne[1];
        const int64_t ne2 = node->ne[2];
        const int64_t nk = ne00 * ne01;
        const int64_t ew0 = nk * ne02;

        size_t cur = 0;

        if (node->src[0]->type == GGML_TYPE_F16 && node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(ggml_fp16_t) * (ne0 * ne1 * ew0);
        } else if (node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * (ne10 * ne11 * ne12);
        } else {
          GGML_ASSERT(false);
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_CONV_TRANSPOSE_2D: {
        n_tasks = n_threads;

        const int64_t ne00 = node->src[0]->ne[0];  // W
        const int64_t ne01 = node->src[0]->ne[1];  // H
        const int64_t ne02 = node->src[0]->ne[2];  // Channels Out
        const int64_t ne03 = node->src[0]->ne[3];  // Channels In

        const int64_t ne10 = node->src[1]->ne[0];  // W
        const int64_t ne11 = node->src[1]->ne[1];  // H
        const int64_t ne12 = node->src[1]->ne[2];  // Channels In

        size_t cur = 0;
        cur += sizeof(ggml_fp16_t) * ne00 * ne01 * ne02 * ne03;
        cur += sizeof(ggml_fp16_t) * ne10 * ne11 * ne12;

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_POOL_1D:
      case GGML_OP_POOL_2D: {
        n_tasks = 1;
      } break;
      case GGML_OP_UPSCALE: {
        n_tasks = n_threads;
      } break;
      case GGML_OP_FLASH_ATTN: {
        n_tasks = n_threads;

        size_t cur = 0;

        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);

        if (node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * ne11 * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * ne11 * n_tasks;  // this is overestimated by x2
        }

        if (node->src[1]->type == GGML_TYPE_F16) {
          cur = sizeof(float) * ne11 * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * ne11 * n_tasks;  // this is overestimated by x2
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_FLASH_FF: {
        n_tasks = n_threads;

        size_t cur = 0;

        if (node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * node->src[1]->ne[1] * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * node->src[1]->ne[1] * n_tasks;  // this is overestimated by x2
        }

        if (node->src[1]->type == GGML_TYPE_F16) {
          cur = sizeof(float) * node->src[1]->ne[1] * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * node->src[1]->ne[1] * n_tasks;  // this is overestimated by x2
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_FLASH_ATTN_BACK: {
        n_tasks = n_threads;

        size_t cur = 0;

        const int64_t D = node->src[0]->ne[0];
        const int64_t ne11 = ggml_up(node->src[1]->ne[1], GGML_SOFT_MAX_UNROLL);
        const int64_t mxDn = MAX(D, ne11) * 2;  // *2 because of S and SM in ggml_compute_forward_flash_attn_back
        if (node->src[1]->type == GGML_TYPE_F32) {
          cur = sizeof(float) * mxDn * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * mxDn * n_tasks;  // this is overestimated by x2
        }

        if (node->src[1]->type == GGML_TYPE_F16) {
          cur = sizeof(float) * mxDn * n_tasks;   // TODO: this can become (n_tasks-1)
          cur += sizeof(float) * mxDn * n_tasks;  // this is overestimated by x2
        }

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_WIN_PART:
      case GGML_OP_WIN_UNPART:
      case GGML_OP_GET_REL_POS:
      case GGML_OP_MAP_UNARY:
      case GGML_OP_MAP_BINARY:
      case GGML_OP_MAP_CUSTOM1_F32:
      case GGML_OP_MAP_CUSTOM2_F32:
      case GGML_OP_MAP_CUSTOM3_F32: {
        n_tasks = 1;
      } break;
      case GGML_OP_MAP_CUSTOM1: {
        struct ggml_map_custom1_op_params* p = (struct ggml_map_custom1_op_params*)node->op_params;
        if (p->n_tasks == GGML_N_TASKS_MAX) {
          n_tasks = n_threads;
        } else {
          n_tasks = MIN(p->n_tasks, n_threads);
        }
      } break;
      case GGML_OP_MAP_CUSTOM2: {
        struct ggml_map_custom2_op_params* p = (struct ggml_map_custom2_op_params*)node->op_params;
        if (p->n_tasks == GGML_N_TASKS_MAX) {
          n_tasks = n_threads;
        } else {
          n_tasks = MIN(p->n_tasks, n_threads);
        }
      } break;
      case GGML_OP_MAP_CUSTOM3: {
        struct ggml_map_custom3_op_params* p = (struct ggml_map_custom3_op_params*)node->op_params;
        if (p->n_tasks == GGML_N_TASKS_MAX) {
          n_tasks = n_threads;
        } else {
          n_tasks = MIN(p->n_tasks, n_threads);
        }
      } break;
      case GGML_OP_CROSS_ENTROPY_LOSS: {
        n_tasks = n_threads;

        size_t cur = type_traits[node->type].type_size * (n_tasks + node->src[0]->ne[0] * n_tasks);

        work_size = MAX(work_size, cur);
      } break;
      case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
        n_tasks = n_threads;
      } break;
      case GGML_OP_NONE: {
        n_tasks = 1;
      } break;
      case GGML_OP_COUNT: {
        GGML_ASSERT(false);
      } break;
    }

    cplan.n_tasks[i] = n_tasks;
  }

  if (work_size > 0) {
    work_size += CACHE_LINE_SIZE * (n_threads - 1);
  }

  cplan.n_threads = n_threads;
  cplan.work_size = work_size;
  cplan.work_data = NULL;

  return cplan;
}

static void ggml_compute_forward_dup_same_cont(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
  GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
  GGML_ASSERT(src0->type == dst->type);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const size_t nb00 = src0->nb[0];
  const size_t nb0 = dst->nb[0];

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  // parallelize by elements
  const int ne = ggml_nelements(dst);
  const int dr = (ne + nth - 1) / nth;
  const int ie0 = dr * ith;
  const int ie1 = MIN(ie0 + dr, ne);

  if (ie0 < ie1) {
    memcpy(((char*)dst->data + ie0 * nb0), ((char*)src0->data + ie0 * nb00),
           (ie1 - ie0) * type_traits[src0->type].type_size);
  }
}
static void ggml_compute_forward_dup_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
    ggml_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }

  // parallelize by rows
  const int nr = ne01;
  // number of rows per thread
  const int dr = (nr + nth - 1) / nth;
  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (src0->type == dst->type && ne00 == ne0 && nb00 == type_traits[src0->type].type_size &&
      nb0 == type_traits[dst->type].type_size) {
    // copy by rows
    const size_t rs = ne00 * nb00;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          memcpy(((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                 ((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03), rs);
        }
      }
    }
    return;
  }

  // TODO: add more special-case implementations for tensor shapes/strides that can benefit from memcpy

  if (ggml_is_contiguous(dst)) {
    if (nb00 == sizeof(ggml_fp16_t)) {
      if (dst->type == GGML_TYPE_F16) {
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        char* dst_ptr = (char*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
              memcpy(dst_ptr + id, src0_ptr, rs);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else if (dst->type == GGML_TYPE_F32) {
        size_t id = 0;
        float* dst_ptr = (float*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
              for (int i00 = 0; i00 < ne00; i00++) {
                dst_ptr[id] = GGML_FP16_TO_FP32(src0_ptr[i00]);
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (type_traits[dst->type].from_float) {
        ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;
        float* src0_f32 = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / type_traits[dst->type].blck_size);
        char* dst_ptr = (char*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

              for (int i00 = 0; i00 < ne00; i00++) {
                src0_f32[i00] = GGML_FP16_TO_FP32(src0_ptr[i00]);
              }

              quantize_row_q(src0_f32, dst_ptr + id, ne00);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else {
        GGML_ASSERT(false);  // TODO: implement
      }
    } else {
      // printf("%s: this is not optimal - fix me\n", __func__);

      if (dst->type == GGML_TYPE_F32) {
        size_t id = 0;
        float* dst_ptr = (float*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const ggml_fp16_t* src0_ptr =
                    (ggml_fp16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = GGML_FP16_TO_FP32(*src0_ptr);
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (dst->type == GGML_TYPE_F16) {
        size_t id = 0;
        ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const ggml_fp16_t* src0_ptr =
                    (ggml_fp16_t*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = *src0_ptr;
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else {
        GGML_ASSERT(false);  // TODO: implement
      }
    }
    return;
  }

  // dst counters
  int64_t i10 = 0;
  int64_t i11 = 0;
  int64_t i12 = 0;
  int64_t i13 = 0;

  if (dst->type == GGML_TYPE_F16) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            memcpy(dst_ptr, src0_ptr, sizeof(ggml_fp16_t));

            if (++i10 == ne00) {
              i10 = 0;
              if (++i11 == ne01) {
                i11 = 0;
                if (++i12 == ne02) {
                  i12 = 0;
                  if (++i13 == ne03) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else if (dst->type == GGML_TYPE_F32) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            *(float*)dst_ptr = GGML_FP16_TO_FP32(*(const ggml_fp16_t*)src0_ptr);

            if (++i10 == ne0) {
              i10 = 0;
              if (++i11 == ne1) {
                i11 = 0;
                if (++i12 == ne2) {
                  i12 = 0;
                  if (++i13 == ne3) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else {
    GGML_ASSERT(false);  // TODO: implement
  }
}

static void ggml_compute_forward_dup_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS

  const int ith = params->ith;  // thread index
  const int nth = params->nth;  // number of threads

  if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
    ggml_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }

  // parallelize by rows
  const int nr = ne01;
  // number of rows per thread
  const int dr = (nr + nth - 1) / nth;
  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (src0->type == dst->type && ne00 == ne0 && nb00 == type_traits[src0->type].type_size &&
      nb0 == type_traits[dst->type].type_size) {
    // copy by rows
    const size_t rs = ne00 * nb00;
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          memcpy(((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                 ((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03), rs);
        }
      }
    }
    return;
  }

  if (ggml_is_contiguous(dst)) {
    // TODO: simplify
    if (nb00 == sizeof(float)) {
      if (dst->type == GGML_TYPE_F32) {
        size_t id = 0;
        const size_t rs = ne00 * nb00;
        char* dst_ptr = (char*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const char* src0_ptr = (char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03;
              memcpy(dst_ptr + id, src0_ptr, rs);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else if (type_traits[dst->type].from_float) {
        ggml_from_float_t const quantize_row_q = type_traits[dst->type].from_float;

        size_t id = 0;
        size_t rs = nb0 * (ne00 / type_traits[dst->type].blck_size);
        char* dst_ptr = (char*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += rs * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              const float* src0_ptr = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
              quantize_row_q(src0_ptr, dst_ptr + id, ne00);
              id += rs;
            }
            id += rs * (ne01 - ir1);
          }
        }
      } else {
        GGML_ASSERT(false);  // TODO: implement
      }
    } else {
      // printf("%s: this is not optimal - fix me\n", __func__);

      if (dst->type == GGML_TYPE_F32) {
        size_t id = 0;
        float* dst_ptr = (float*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = *src0_ptr;
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else if (dst->type == GGML_TYPE_F16) {
        size_t id = 0;
        ggml_fp16_t* dst_ptr = (ggml_fp16_t*)dst->data;

        for (int i03 = 0; i03 < ne03; i03++) {
          for (int i02 = 0; i02 < ne02; i02++) {
            id += ne00 * ir0;
            for (int i01 = ir0; i01 < ir1; i01++) {
              for (int i00 = 0; i00 < ne00; i00++) {
                const float* src0_ptr = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

                dst_ptr[id] = GGML_FP32_TO_FP16(*src0_ptr);
                id++;
              }
            }
            id += ne00 * (ne01 - ir1);
          }
        }
      } else {
        GGML_ASSERT(false);  // TODO: implement
      }
    }

    return;
  }

  // dst counters

  int64_t i10 = 0;
  int64_t i11 = 0;
  int64_t i12 = 0;
  int64_t i13 = 0;

  if (dst->type == GGML_TYPE_F32) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            memcpy(dst_ptr, src0_ptr, sizeof(float));

            if (++i10 == ne0) {
              i10 = 0;
              if (++i11 == ne1) {
                i11 = 0;
                if (++i12 == ne2) {
                  i12 = 0;
                  if (++i13 == ne3) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else if (dst->type == GGML_TYPE_F16) {
    for (int64_t i03 = 0; i03 < ne03; i03++) {
      for (int64_t i02 = 0; i02 < ne02; i02++) {
        i10 += ne00 * ir0;
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
        for (int64_t i01 = ir0; i01 < ir1; i01++) {
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            const char* src0_ptr = ((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);
            char* dst_ptr = ((char*)dst->data + i10 * nb0 + i11 * nb1 + i12 * nb2 + i13 * nb3);

            *(ggml_fp16_t*)dst_ptr = GGML_FP32_TO_FP16(*(const float*)src0_ptr);

            if (++i10 == ne0) {
              i10 = 0;
              if (++i11 == ne1) {
                i11 = 0;
                if (++i12 == ne2) {
                  i12 = 0;
                  if (++i13 == ne3) {
                    i13 = 0;
                  }
                }
              }
            }
          }
        }
        i10 += ne00 * (ne01 - ir1);
        while (i10 >= ne0) {
          i10 -= ne0;
          if (++i11 == ne1) {
            i11 = 0;
            if (++i12 == ne2) {
              i12 = 0;
              if (++i13 == ne3) {
                i13 = 0;
              }
            }
          }
        }
      }
    }
  } else {
    GGML_ASSERT(false);  // TODO: implement
  }
}

static void ggml_compute_forward_dup(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  if (ggml_is_contiguous(src0) && ggml_is_contiguous(dst) && src0->type == dst->type) {
    ggml_compute_forward_dup_same_cont(params, src0, dst);
    return;
  }
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_dup_f16(params, src0, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_dup_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_add
static void ggml_compute_forward_add_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_can_repeat_rows(src1, src0) && ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (nb10 == sizeof(float)) {
    for (int ir = ir0; ir < ir1; ++ir) {
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
      float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

#ifdef GGML_USE_ACCELERATE
      vDSP_vadd(src0_ptr, 1, src1_ptr, 1, dst_ptr, 1, ne00);
#else
      ggml_vec_add_f32(ne00, dst_ptr, src0_ptr, src1_ptr);
#endif
    }
  } else {
    // src1 is not contiguous
    for (int ir = ir0; ir < ir1; ++ir) {
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

      for (int i0 = 0; i0 < ne0; i0++) {
        float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] + *src1_ptr;
      }
    }
  }
}

static void ggml_compute_forward_add_f16_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                             const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F16);

  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (nb10 == sizeof(float)) {
    for (int ir = ir0; ir < ir1; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

      for (int i = 0; i < ne0; i++) {
        dst_ptr[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src0_ptr[i]) + src1_ptr[i]);
      }
    }
  } else {
    // src1 is not contiguous
    GGML_ASSERT(false);
  }
}

static void ggml_compute_forward_add_f16_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                             const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F16);
  GGML_ASSERT(dst->type == GGML_TYPE_F16);

  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  if (nb10 == sizeof(ggml_fp16_t)) {
    for (int ir = ir0; ir < ir1; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      ggml_fp16_t* src1_ptr = (ggml_fp16_t*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11);

      for (int i = 0; i < ne0; i++) {
        dst_ptr[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src0_ptr[i]) + GGML_FP16_TO_FP32(src1_ptr[i]));
      }
    }
  } else {
    // src1 is not contiguous
    GGML_ASSERT(false);
  }
}

static void ggml_compute_forward_add_q_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                           const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  const enum ggml_type type = src0->type;
  const enum ggml_type dtype = dst->type;
  ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;
  ggml_from_float_t const quantize_row_q = type_traits[dtype].from_float;

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == type_traits[type].type_size);
  GGML_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(type_traits[src0->type].is_quantized);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  float* wdata = (float*)params->wdata + (ne00 + CACHE_LINE_SIZE_F32) * ith;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 indices
    const int i03 = ir / (ne02 * ne01);
    const int i02 = (ir - i03 * ne02 * ne01) / ne01;
    const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

    // src1 and dst are same shape as src0 => same indices
    const int i13 = i03;
    const int i12 = i02;
    const int i11 = i01;

    const int i3 = i03;
    const int i2 = i02;
    const int i1 = i01;

    void* src0_row = (void*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
    float* src1_row = (float*)((char*)src1->data + (i11 * nb11 + i12 * nb12 + i13 * nb13));
    void* dst_row = (void*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

    assert(ne00 % 32 == 0);

    // unquantize row from src0 to temp buffer
    dequantize_row_q(src0_row, wdata, ne00);
    // add src1
    ggml_vec_acc_f32(ne00, wdata, src1_row);
    // quantize row to dst
    if (quantize_row_q != NULL) {
      quantize_row_q(wdata, dst_row, ne00);
    } else {
      memcpy(dst_row, wdata, ne0 * nb0);
    }
  }
}

static void ggml_compute_forward_add(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_add_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16: {
      if (src1->type == GGML_TYPE_F16) {
        ggml_compute_forward_add_f16_f16(params, src0, src1, dst);
      } else if (src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_add_f16_f32(params, src0, src1, dst);
      } else {
        GGML_ASSERT(false);
      }
    } break;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: {
      ggml_compute_forward_add_q_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_add1

static void ggml_compute_forward_add1_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(src1));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

#ifdef GGML_USE_ACCELERATE
    // UNUSED(ggml_vec_add1_f32);

    vDSP_vadd((float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), 1, (float*)((char*)src1->data), 0,
              (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1), 1, ne0);
#else
    ggml_vec_add1_f32(ne0, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1),
                      (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), *(float*)src1->data);
#endif
  }
}

static void ggml_compute_forward_add1_f16_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(src1));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = *(float*)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F16);

  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
    ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
    for (int i = 0; i < ne0; i++) {
      dst_ptr[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src0_ptr[i]) + v);
    }
  }
}

static void ggml_compute_forward_add1_f16_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(src1));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = GGML_FP16_TO_FP32(*(ggml_fp16_t*)src1->data);

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F16);
  GGML_ASSERT(dst->type == GGML_TYPE_F16);

  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    ggml_fp16_t* dst_ptr = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
    ggml_fp16_t* src0_ptr = (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
    for (int i = 0; i < ne0; i++) {
      dst_ptr[i] = GGML_FP32_TO_FP16(GGML_FP16_TO_FP32(src0_ptr[i]) + v);
    }
  }
}

static void ggml_compute_forward_add1_q_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(src1));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // scalar to add
  const float v = *(float*)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_UNARY_OP_LOCALS

  const enum ggml_type type = src0->type;
  ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;
  ggml_from_float_t const quantize_row_q = type_traits[type].from_float;

  // we don't support permuted src0
  GGML_ASSERT(nb00 == type_traits[type].type_size);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(type_traits[src0->type].is_quantized);
  GGML_ASSERT(dst->type == src0->type);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  float* wdata = (float*)params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are same shape => same indices
    const int i3 = ir / (ne2 * ne1);
    const int i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    void* src0_row = (void*)((char*)src0->data + (i1 * nb01 + i2 * nb02 + i3 * nb03));
    void* dst_row = (void*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb0));

    assert(ne0 % 32 == 0);

    // unquantize row from src0 to temp buffer
    dequantize_row_q(src0_row, wdata, ne0);
    // add src1
    ggml_vec_acc1_f32(ne0, wdata, v);
    // quantize row to dst
    quantize_row_q(wdata, dst_row, ne0);
  }
}

static void ggml_compute_forward_add1(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_add1_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16: {
      if (src1->type == GGML_TYPE_F16) {
        ggml_compute_forward_add1_f16_f16(params, src0, src1, dst);
      } else if (src1->type == GGML_TYPE_F32) {
        ggml_compute_forward_add1_f16_f32(params, src0, src1, dst);
      } else {
        GGML_ASSERT(false);
      }
    } break;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: {
      ggml_compute_forward_add1_q_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_acc

static void ggml_compute_forward_acc_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

  // view src0 and dst with these strides and data offset inbytes during acc
  // nb0 is implicitely element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t*)dst->op_params)[0];
  size_t nb2 = ((int32_t*)dst->op_params)[1];
  size_t nb3 = ((int32_t*)dst->op_params)[2];
  size_t offset = ((int32_t*)dst->op_params)[3];
  bool inplace = (bool)((int32_t*)dst->op_params)[4];

  if (!inplace && (params->type == GGML_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase
    memcpy(((char*)dst->data), ((char*)src0->data), ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src1);
  const int nc = src1->ne[0];

  GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
  GGML_TENSOR_LOCALS(size_t, nb1, src1, nb)

  // src0 and dst as viewed during acc
  const size_t nb0 = type_traits[src0->type].type_size;

  const size_t nb00 = nb0;
  const size_t nb01 = nb1;
  const size_t nb02 = nb2;
  const size_t nb03 = nb3;

  GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10 - 1) * nb0 + (ne11 == 0 ? 0 : ne11 - 1) * nb1 +
                  (ne12 == 0 ? 0 : ne12 - 1) * nb2 + (ne13 == 0 ? 0 : ne13 - 1) * nb3 <
              ggml_nbytes(dst));
  GGML_ASSERT(offset + (ne10 == 0 ? 0 : ne10 - 1) * nb00 + (ne11 == 0 ? 0 : ne11 - 1) * nb01 +
                  (ne12 == 0 ? 0 : ne12 - 1) * nb02 + (ne13 == 0 ? 0 : ne13 - 1) * nb03 <
              ggml_nbytes(src0));

  GGML_ASSERT(nb10 == sizeof(float));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int i3 = ir / (ne12 * ne11);
    const int i2 = (ir - i3 * ne12 * ne11) / ne11;
    const int i1 = (ir - i3 * ne12 * ne11 - i2 * ne11);

#ifdef GGML_USE_ACCELERATE
    vDSP_vadd((float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + offset), 1,
              (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11), 1,
              (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset), 1, nc);
#else
    ggml_vec_add_f32(nc, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset),
                     (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + offset),
                     (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
#endif
  }
}

static void ggml_compute_forward_acc(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_acc_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_sub

static void ggml_compute_forward_sub_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  if (nb10 == sizeof(float)) {
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

#ifdef GGML_USE_ACCELERATE
      vDSP_vsub((float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11), 1,
                (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), 1,
                (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1), 1, ne0);
#else
      ggml_vec_sub_f32(ne0, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1),
                       (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01),
                       (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
#endif
      // }
      // }
    }
  } else {
    // src1 is not contiguous
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      float* dst_ptr = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      for (int i0 = 0; i0 < ne0; i0++) {
        float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] - *src1_ptr;
      }
    }
  }
}

static void ggml_compute_forward_sub(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sub_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_mul

static void ggml_compute_forward_mul_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_can_repeat_rows(src1, src0) && ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }
  const int ith = params->ith;
  const int nth = params->nth;

#ifdef GGML_USE_CLBLAST
  if (src1->backend == GGML_BACKEND_GPU) {
    if (ith == 0) {
      ggml_cl_mul(src0, src1, dst);
    }
    return;
  }
#endif

  const int64_t nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(ne00 == ne10);

  if (nb10 == sizeof(float)) {
    for (int64_t ir = ith; ir < nr; ir += nth) {
      // src0 and dst are same shape => same indices
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);
      float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11);

#ifdef GGML_USE_ACCELERATE
      // UNUSED(ggml_vec_mul_f32);

      vDSP_vmul(src0_ptr, 1, src1_ptr, 1, dst_ptr, 1, ne00);
#else
      ggml_vec_mul_f32(ne00, dst_ptr, src0_ptr, src1_ptr);
#endif
      // }
      // }
    }
  } else {
    // src1 is not contiguous
    for (int64_t ir = ith; ir < nr; ir += nth) {
      // src0 and dst are same shape => same indices
      // src1 is broadcastable across src0 and dst in i1, i2, i3
      const int64_t i03 = ir / (ne02 * ne01);
      const int64_t i02 = (ir - i03 * ne02 * ne01) / ne01;
      const int64_t i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);

      const int64_t i13 = i03 % ne13;
      const int64_t i12 = i02 % ne12;
      const int64_t i11 = i01 % ne11;

      float* dst_ptr = (float*)((char*)dst->data + i03 * nb3 + i02 * nb2 + i01 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i03 * nb03 + i02 * nb02 + i01 * nb01);

      for (int64_t i0 = 0; i0 < ne00; i0++) {
        float* src1_ptr = (float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] * (*src1_ptr);
      }
    }
  }
}

static void ggml_compute_forward_mul(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(src1->type == GGML_TYPE_F32 && "only f32 src1 supported for now");

  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_mul_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_div

static void ggml_compute_forward_div_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nr = ggml_nrows(src0);

  GGML_TENSOR_BINARY_OP_LOCALS

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  if (nb10 == sizeof(float)) {
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

#ifdef GGML_USE_ACCELERATE
      // UNUSED(ggml_vec_div_f32);

      vDSP_vdiv((float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11), 1,
                (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01), 1,
                (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1), 1, ne0);
#else
      ggml_vec_div_f32(ne0, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1),
                       (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01),
                       (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
#endif
      // }
      // }
    }
  } else {
    // src1 is not contiguous
    for (int ir = 0; ir < nr; ++ir) {
      // src0, src1 and dst are same shape => same indices
      const int i3 = ir / (ne2 * ne1);
      const int i2 = (ir - i3 * ne2 * ne1) / ne1;
      const int i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

      float* dst_ptr = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
      float* src0_ptr = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01);
      for (int i0 = 0; i0 < ne0; i0++) {
        float* src1_ptr = (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);

        dst_ptr[i0] = src0_ptr[i0] / (*src1_ptr);
      }
    }
  }
}

static void ggml_compute_forward_div(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_div_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_sqr

static void ggml_compute_forward_sqr_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_sqr_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                     (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_sqr(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sqr_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_sqrt

static void ggml_compute_forward_sqrt_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_sqrt_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                      (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_sqrt(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sqrt_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_log

static void ggml_compute_forward_log_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  GGML_ASSERT(params->ith == 0);
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  GGML_ASSERT(dst->nb[0] == sizeof(float));
  GGML_ASSERT(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_log_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                     (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_log(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_log_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_sum

static void ggml_compute_forward_sum_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_is_scalar(dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  assert(ggml_is_scalar(dst));
  assert(src0->nb[0] == sizeof(float));

  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)

  ggml_float sum = 0;
  ggml_float row_sum = 0;

  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = 0; i01 < ne01; i01++) {
        ggml_vec_sum_f32_ggf(ne00, &row_sum, (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
        sum += row_sum;
      }
    }
  }
  ((float*)dst->data)[0] = sum;
}

static void ggml_compute_forward_sum_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_is_scalar(dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  assert(src0->nb[0] == sizeof(ggml_fp16_t));

  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
  GGML_TENSOR_LOCALS(size_t, nb0, src0, nb)

  float sum = 0;
  float row_sum = 0;

  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = 0; i01 < ne01; i01++) {
        ggml_vec_sum_f16_ggf(ne00, &row_sum, (ggml_fp16_t*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));
        sum += row_sum;
      }
    }
  }
  ((ggml_fp16_t*)dst->data)[0] = GGML_FP32_TO_FP16(sum);
}

static void ggml_compute_forward_sum(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sum_f32(params, src0, dst);
    } break;
    case GGML_TYPE_F16: {
      ggml_compute_forward_sum_f16(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_sum_rows

static void ggml_compute_forward_sum_rows_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              struct ggml_tensor* dst) {
  GGML_ASSERT(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_ASSERT(src0->nb[0] == sizeof(float));
  GGML_ASSERT(dst->nb[0] == sizeof(float));

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(ne0 == 1);
  GGML_ASSERT(ne1 == ne01);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  for (int64_t i3 = 0; i3 < ne03; i3++) {
    for (int64_t i2 = 0; i2 < ne02; i2++) {
      for (int64_t i1 = 0; i1 < ne01; i1++) {
        float* src_row = (float*)((char*)src0->data + i1 * nb01 + i2 * nb02 + i3 * nb03);
        float* dst_row = (float*)((char*)dst->data + i1 * nb1 + i2 * nb2 + i3 * nb3);
        float row_sum = 0;
        ggml_vec_sum_f32(ne00, &row_sum, src_row);
        dst_row[0] = row_sum;
      }
    }
  }
}

static void ggml_compute_forward_sum_rows(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sum_rows_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_mean

static void ggml_compute_forward_mean_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  assert(src0->nb[0] == sizeof(float));

  GGML_TENSOR_UNARY_OP_LOCALS

  assert(ne0 == 1);
  assert(ne1 == ne01);
  assert(ne2 == ne02);
  assert(ne3 == ne03);

  // UNUSED(ne0);
  // UNUSED(ne1);
  // UNUSED(ne2);
  // UNUSED(ne3);

  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = 0; i01 < ne01; i01++) {
        ggml_vec_sum_f32(ne00, (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3),
                         (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03));

        *(float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3) /= (float)ne00;
      }
    }
  }
}

static void ggml_compute_forward_mean(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_mean_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_argmax

static void ggml_compute_forward_argmax_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            struct ggml_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  assert(src0->nb[0] == sizeof(float));
  assert(dst->nb[0] == sizeof(float));

  const int64_t ne00 = src0->ne[0];
  const int64_t ne01 = src0->ne[1];

  const size_t nb01 = src0->nb[1];
  const size_t nb0 = dst->nb[0];

  for (int64_t i1 = 0; i1 < ne01; i1++) {
    float* src = (float*)((char*)src0->data + i1 * nb01);
    int32_t* dst_ = (int32_t*)((char*)dst->data + i1 * nb0);
    int v = 0;
    ggml_vec_argmax_f32(ne00, &v, src);
    dst_[0] = v;
  }
}

static void ggml_compute_forward_argmax(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                        struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_argmax_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_repeat

static void ggml_compute_forward_repeat_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            struct ggml_tensor* dst) {
  GGML_ASSERT(params->ith == 0);
  GGML_ASSERT(ggml_can_repeat(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int nr0 = (int)(ne0 / ne00);
  const int nr1 = (int)(ne1 / ne01);
  const int nr2 = (int)(ne2 / ne02);
  const int nr3 = (int)(ne3 / ne03);

  // TODO: support for transposed / permuted tensors
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  // TODO: maybe this is not optimal?
  for (int i3 = 0; i3 < nr3; i3++) {
    for (int k3 = 0; k3 < ne03; k3++) {
      for (int i2 = 0; i2 < nr2; i2++) {
        for (int k2 = 0; k2 < ne02; k2++) {
          for (int i1 = 0; i1 < nr1; i1++) {
            for (int k1 = 0; k1 < ne01; k1++) {
              for (int i0 = 0; i0 < nr0; i0++) {
                ggml_vec_cpy_f32(ne00,
                                 (float*)((char*)dst->data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 +
                                          (i1 * ne01 + k1) * nb1 + (i0 * ne00) * nb0),
                                 (float*)((char*)src0->data + (k3)*nb03 + (k2)*nb02 + (k1)*nb01));
              }
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_repeat_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            struct ggml_tensor* dst) {
  GGML_ASSERT(params->ith == 0);
  GGML_ASSERT(ggml_can_repeat(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS;

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int nr0 = (int)(ne0 / ne00);
  const int nr1 = (int)(ne1 / ne01);
  const int nr2 = (int)(ne2 / ne02);
  const int nr3 = (int)(ne3 / ne03);

  // TODO: support for transposed / permuted tensors
  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));

  // TODO: maybe this is not optimal?
  for (int i3 = 0; i3 < nr3; i3++) {
    for (int k3 = 0; k3 < ne03; k3++) {
      for (int i2 = 0; i2 < nr2; i2++) {
        for (int k2 = 0; k2 < ne02; k2++) {
          for (int i1 = 0; i1 < nr1; i1++) {
            for (int k1 = 0; k1 < ne01; k1++) {
              for (int i0 = 0; i0 < nr0; i0++) {
                ggml_fp16_t* y = (ggml_fp16_t*)((char*)dst->data + (i3 * ne03 + k3) * nb3 + (i2 * ne02 + k2) * nb2 +
                                                (i1 * ne01 + k1) * nb1 + (i0 * ne00) * nb0);
                ggml_fp16_t* x = (ggml_fp16_t*)((char*)src0->data + (k3)*nb03 + (k2)*nb02 + (k1)*nb01);
                // ggml_vec_cpy_f16(ne00, y, x)
                for (int i = 0; i < ne00; ++i) {
                  y[i] = x[i];
                }
              }
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_repeat(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                        struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_repeat_f16(params, src0, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_repeat_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_repeat_back

static void ggml_compute_forward_repeat_back_f32(const struct ggml_compute_params* params,
                                                 const struct ggml_tensor* src0, struct ggml_tensor* dst) {
  GGML_ASSERT(params->ith == 0);
  GGML_ASSERT(ggml_can_repeat(dst, src0));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_UNARY_OP_LOCALS

  // guaranteed to be an integer due to the check in ggml_can_repeat
  const int nr0 = (int)(ne00 / ne0);
  const int nr1 = (int)(ne01 / ne1);
  const int nr2 = (int)(ne02 / ne2);
  const int nr3 = (int)(ne03 / ne3);

  // TODO: support for transposed / permuted tensors
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  if (ggml_is_contiguous(dst)) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float*)dst->data, 0);
  } else {
    for (int k3 = 0; k3 < ne3; k3++) {
      for (int k2 = 0; k2 < ne2; k2++) {
        for (int k1 = 0; k1 < ne1; k1++) {
          ggml_vec_set_f32(ne0, (float*)((char*)dst->data + k1 * nb1 + k2 * nb2 + k3 * nb3), 0);
        }
      }
    }
  }

  // TODO: maybe this is not optimal?
  for (int i3 = 0; i3 < nr3; i3++) {
    for (int k3 = 0; k3 < ne3; k3++) {
      for (int i2 = 0; i2 < nr2; i2++) {
        for (int k2 = 0; k2 < ne2; k2++) {
          for (int i1 = 0; i1 < nr1; i1++) {
            for (int k1 = 0; k1 < ne1; k1++) {
              for (int i0 = 0; i0 < nr0; i0++) {
                ggml_vec_acc_f32(ne0, (float*)((char*)dst->data + (k3)*nb3 + (k2)*nb2 + (k1)*nb1),
                                 (float*)((char*)src0->data + (i3 * ne3 + k3) * nb03 + (i2 * ne2 + k2) * nb02 +
                                          (i1 * ne1 + k1) * nb01 + (i0 * ne0) * nb00));
              }
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_repeat_back(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                             struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_repeat_back_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_concat

static void ggml_compute_forward_concat_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;

  GGML_TENSOR_BINARY_OP_LOCALS

  // TODO: support for transposed / permuted tensors
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));

  for (int i3 = 0; i3 < ne3; i3++) {
    for (int i2 = ith; i2 < ne2; i2++) {
      if (i2 < ne02) {  // src0
        for (int i1 = 0; i1 < ne1; i1++) {
          for (int i0 = 0; i0 < ne0; i0++) {
            const float* x = (float*)((char*)src0->data + i0 * nb00 + i1 * nb01 + i2 * nb02 + i3 * nb03);

            float* y = (float*)((char*)dst->data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
            *y = *x;
          }
        }
      }  // src1
      else {
        for (int i1 = 0; i1 < ne1; i1++) {
          for (int i0 = 0; i0 < ne0; i0++) {
            const float* x = (float*)((char*)src1->data + i0 * nb10 + i1 * nb11 + (i2 - ne02) * nb12 + i3 * nb13);

            float* y = (float*)((char*)dst->data + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
            *y = *x;
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_concat(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                        const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_concat_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_abs

static void ggml_compute_forward_abs_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_abs_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                     (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_abs(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_abs_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_sgn

static void ggml_compute_forward_sgn_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_sgn_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                     (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_sgn(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_sgn_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_neg

static void ggml_compute_forward_neg_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_neg_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                     (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_neg(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_neg_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_step

static void ggml_compute_forward_step_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_step_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                      (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_step(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_step_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_tanh

static void ggml_compute_forward_tanh_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_tanh_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                      (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_tanh(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_tanh_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_elu

static void ggml_compute_forward_elu_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_elu_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                     (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_elu(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_elu_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_relu

static void ggml_compute_forward_relu_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    ggml_vec_relu_f32(nc, (float*)((char*)dst->data + i * (dst->nb[1])),
                      (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_relu(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_relu_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_gelu

static void ggml_compute_forward_gelu_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(src0));
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(dst));
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ggml_vec_gelu_f32(nc, (float*)((char*)dst->data + i1 * (dst->nb[1])),
                      (float*)((char*)src0->data + i1 * (src0->nb[1])));

#ifndef NDEBUG
    for (int k = 0; k < nc; k++) {
      const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
      // UNUSED(x);
      assert(!isnan(x));
      assert(!isinf(x));
    }
#endif
  }
}

static void ggml_compute_forward_gelu(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_gelu_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_gelu_quick

static void ggml_compute_forward_gelu_quick_f32(const struct ggml_compute_params* params,
                                                const struct ggml_tensor* src0, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(src0));
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(dst));
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ggml_vec_gelu_quick_f32(nc, (float*)((char*)dst->data + i1 * (dst->nb[1])),
                            (float*)((char*)src0->data + i1 * (src0->nb[1])));

#ifndef NDEBUG
    for (int k = 0; k < nc; k++) {
      const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
      // UNUSED(x);
      assert(!isnan(x));
      assert(!isinf(x));
    }
#endif
  }
}

static void ggml_compute_forward_gelu_quick(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_gelu_quick_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_silu

static void ggml_compute_forward_silu_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(src0));
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(dst));
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ggml_vec_silu_f32(nc, (float*)((char*)dst->data + i1 * (dst->nb[1])),
                      (float*)((char*)src0->data + i1 * (src0->nb[1])));

#ifndef NDEBUG
    for (int k = 0; k < nc; k++) {
      const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
      // UNUSED(x);
      assert(!isnan(x));
      assert(!isinf(x));
    }
#endif
  }
}

static void ggml_compute_forward_silu(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_silu_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_silu_back

static void ggml_compute_forward_silu_back_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               const struct ggml_tensor* grad, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(grad));
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(src0));
  GGML_ASSERT(ggml_is_contiguous_except_dim_1(dst));
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_are_same_shape(src0, grad));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    ggml_vec_silu_backward_f32(nc, (float*)((char*)dst->data + i1 * (dst->nb[1])),
                               (float*)((char*)src0->data + i1 * (src0->nb[1])),
                               (float*)((char*)grad->data + i1 * (grad->nb[1])));

#ifndef NDEBUG
    for (int k = 0; k < nc; k++) {
      const float x = ((float*)((char*)dst->data + i1 * (dst->nb[1])))[k];
      // UNUSED(x);
      assert(!isnan(x));
      assert(!isinf(x));
    }
#endif
  }
}

static void ggml_compute_forward_silu_back(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                           const struct ggml_tensor* grad, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_silu_back_f32(params, src0, grad, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_norm

static void ggml_compute_forward_norm_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  float eps;
  memcpy(&eps, dst->op_params, sizeof(float));

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

        ggml_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum += (ggml_float)x[i00];
        }

        float mean = sum / ne00;

        float* y = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        ggml_float sum2 = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          float v = x[i00] - mean;
          y[i00] = v;
          sum2 += (ggml_float)(v * v);
        }

        float variance = sum2 / ne00;
        const float scale = 1.0f / sqrtf(variance + eps);

        ggml_vec_scale_f32(ne00, y, scale);
      }
    }
  }
}

static void ggml_compute_forward_norm(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_norm_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_group_rms_norm

static void ggml_compute_forward_rms_norm_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  float eps;
  memcpy(&eps, dst->op_params, sizeof(float));

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

        ggml_float sum = 0.0;
        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum += (ggml_float)(x[i00] * x[i00]);
        }

        const float mean = sum / ne00;

        float* y = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        memcpy(y, x, ne00 * sizeof(float));
        // for (int i00 = 0; i00 < ne00; i00++) {
        //     y[i00] = x[i00];
        // }

        const float scale = 1.0f / sqrtf(mean + eps);

        ggml_vec_scale_f32(ne00, y, scale);
      }
    }
  }
}

static void ggml_compute_forward_rms_norm(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_rms_norm_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

static void ggml_compute_forward_rms_norm_back_f32(const struct ggml_compute_params* params,
                                                   const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                   struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst) && ggml_are_same_shape(src0, src1));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  GGML_TENSOR_BINARY_OP_LOCALS

  float eps;
  memcpy(&eps, dst->op_params, sizeof(float));

  // TODO: optimize
  for (int64_t i03 = 0; i03 < ne03; i03++) {
    for (int64_t i02 = 0; i02 < ne02; i02++) {
      for (int64_t i01 = ith; i01 < ne01; i01 += nth) {
        // src1 is same shape as src0 => same indices
        const int64_t i11 = i01;
        const int64_t i12 = i02;
        const int64_t i13 = i03;

        const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);
        const float* dz = (float*)((char*)src1->data + i11 * nb11 + i12 * nb12 + i13 * nb13);

        ggml_float sum_xx = 0.0;
        ggml_float sum_xdz = 0.0;

        for (int64_t i00 = 0; i00 < ne00; i00++) {
          sum_xx += (ggml_float)(x[i00] * x[i00]);
          sum_xdz += (ggml_float)(x[i00] * dz[i00]);
        }

        // const float mean     = (float)(sum_xx)/ne00;
        const float mean_eps = (float)(sum_xx) / ne00 + eps;
        const float sum_eps = (float)(sum_xx) + eps * ne00;
        // const float mean_xdz = (float)(sum_xdz)/ne00;
        //  we could cache rms from forward pass to improve performance.
        //  to do this implement ggml_rms and compose ggml_rms_norm using ggml_rms.
        // const float rms      = sqrtf(mean_eps);
        const float rrms = 1.0f / sqrtf(mean_eps);
        // const float scale    = -rrms/(ne00 * mean_eps); // -1/(n*rms**3)

        {
          // z = rms_norm(x)
          //
          // rms_norm(src0) =
          //     scale(
          //         src0,
          //         div(
          //             1,
          //             sqrt(
          //                 add(
          //                     scale(
          //                         sum(
          //                             sqr(
          //                                 src0)),
          //                         (1.0/N)),
          //                     eps))));

          // postorder:
          // ## op    args         grad
          // 00 param src0         grad[#00]
          // 01 const 1
          // 02 sqr   (#00)        grad[#02]
          // 03 sum   (#02)        grad[#03]
          // 04 const 1/N
          // 05 scale (#03, #04)   grad[#05]
          // 06 const eps
          // 07 add   (#05, #06)   grad[#07]
          // 08 sqrt  (#07)        grad[#08]
          // 09 div   (#01,#08)    grad[#09]
          // 10 scale (#00,#09)    grad[#10]
          //
          // backward pass, given grad[#10]
          // #10: scale
          // grad[#00] += scale(grad[#10],#09)
          // grad[#09] += sum(mul(grad[#10],#00))
          // #09: div
          // grad[#08] += neg(mul(grad[#09], div(#09,#08)))
          // #08: sqrt
          // grad[#07] += mul(grad[#08], div(0.5, #08))
          // #07: add
          // grad[#05] += grad[#07]
          // #05: scale
          // grad[#03] += scale(grad[#05],#04)
          // #03: sum
          // grad[#02] += repeat(grad[#03], #02)
          // #02:
          // grad[#00] += scale(mul(#00, grad[#02]), 2.0)
          //
          // substitute and simplify:
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
          // grad[#02] = repeat(grad[#03], #02)
          // grad[#02] = repeat(scale(grad[#05],#04), #02)
          // grad[#02] = repeat(scale(grad[#07],#04), #02)
          // grad[#02] = repeat(scale(mul(grad[#08], div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(scale(mul(neg(mul(grad[#09], div(#09,#08))), div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(scale(mul(neg(mul(sum(mul(grad[#10],#00)), div(#09,#08))), div(0.5, #08)),#04), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(#09,#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(div(#01,#08),#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#08*#08) * div(0.5, #08) * (1/N)), #02)
          // grad[#02] = repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5, #08) * (1/N)), #02)
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, grad[#02]), 2.0)
          // grad[#00] = scale(grad(#10), #09) + scale(mul(#00, repeat(-(sum(mul(grad[#10],#00)) * div(1,#07) * div(0.5,
          // #08) * (1/N)), #02)), 2.0) grad[#00] = scale(grad(#10), #09) + scale(scale(#00, -(sum(mul(grad[#10],#00)) *
          // div(1,#07) * div(0.5, #08) * (1/N))), 2.0) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // -(sum(mul(grad[#10],#00)) * div(1,#07) * div(1,#08) * (1/N))) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,#07*#08) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(1,mean_eps*rms) * (-1/N)) grad[#00] = scale(grad(#10), #09) +
          // scale(#00, sum(mul(grad[#10],#00)) * div(-1,rms*N*mean_eps)) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // sum(mul(grad[#10],#00)) * div(-1,rms*N*(sum_xx/N+eps))) grad[#00] = scale(grad(#10), #09) + scale(#00,
          // sum(mul(grad[#10],#00)) * div(-1,rms*N*sum_xx+rms*N*eps)) grad[#00] = scale(dz, rrms) + scale(x,
          // sum(mul(dz,x)) * div(-1,rms*N*mean_eps)) grad[#00] = scale(dz, rrms) + scale(x, sum_xdz *
          // div(-1,rms*N*mean_eps)) a = b*c + d*e a = b*c*f/f + d*e*f/f a = (b*c*f + d*e*f)*(1/f) a = (b*c*(1/c) +
          // d*e*(1/c))*(1/(1/c)) a = (b + d*e/c)*c b = dz, c = rrms, d = x, e = sum_xdz * div(-1,rms*N*mean_eps) a =
          // (dz + x*sum_xdz * div(-1,rms*N*mean_eps)/rrms)*rrms a = (dz + x*sum_xdz * div(-1,rms*N*mean_eps)*rms)*rrms
          // a = (dz + x*sum_xdz * div(-rms,rms*N*mean_eps))*rrms
          // a = (dz + x*sum_xdz * div(-1,N*mean_eps))*rrms
          // a = (dz + x*div(-sum_xdz,N*mean_eps))*rrms
          // a = (dz + x*div(-mean_xdz,mean_eps))*rrms
          // grad[#00] = scale(dz + scale(x, div(-mean_xdz,mean_eps)),rrms)
          // grad[#00] = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
          // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
        }
        // dx = scale(dz + scale(x, -mean_xdz/mean_eps),rrms)
        // post-order:
        // dx := x
        // dx := scale(dx,-mean_xdz/mean_eps)
        // dx := add(dx, dz)
        // dx := scale(dx, rrms)
        float* dx = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

        ggml_vec_cpy_f32(ne00, dx, x);
        // ggml_vec_scale_f32(ne00, dx, -mean_xdz/mean_eps);
        ggml_vec_scale_f32(ne00, dx, (float)(-sum_xdz) / sum_eps);
        ggml_vec_acc_f32(ne00, dx, dz);
        ggml_vec_scale_f32(ne00, dx, rrms);
      }
    }
  }
}

static void ggml_compute_forward_rms_norm_back(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_rms_norm_back_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_group_norm

static void ggml_compute_forward_group_norm_f32(const struct ggml_compute_params* params,
                                                const struct ggml_tensor* src0, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  GGML_TENSOR_UNARY_OP_LOCALS

  const float eps = 1e-6f;  // TODO: make this a parameter

  // TODO: optimize

  int n_channels = src0->ne[2];
  int n_groups = dst->op_params[0];
  int n_channels_per_group = (n_channels + n_groups - 1) / n_groups;
  for (int i = ith; i < n_groups; i += nth) {
    int start = i * n_channels_per_group;
    int end = start + n_channels_per_group;
    if (end > n_channels) {
      end = n_channels;
    }
    int step = end - start;

    for (int64_t i03 = 0; i03 < ne03; i03++) {
      ggml_float sum = 0.0;
      for (int64_t i02 = start; i02 < end; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

          for (int64_t i00 = 0; i00 < ne00; i00++) {
            sum += (ggml_float)x[i00];
          }
        }
      }
      float mean = sum / (ne00 * ne01 * step);
      ggml_float sum2 = 0.0;

      for (int64_t i02 = start; i02 < end; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* x = (float*)((char*)src0->data + i01 * nb01 + i02 * nb02 + i03 * nb03);

          float* y = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);

          for (int64_t i00 = 0; i00 < ne00; i00++) {
            float v = x[i00] - mean;
            y[i00] = v;
            sum2 += (ggml_float)(v * v);
          }
        }
      }
      float variance = sum2 / (ne00 * ne01 * step);
      const float scale = 1.0f / sqrtf(variance + eps);

      for (int64_t i02 = start; i02 < end; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          float* y = (float*)((char*)dst->data + i01 * nb1 + i02 * nb2 + i03 * nb3);
          ggml_vec_scale_f32(ne00, y, scale);
        }
      }
    }
  }
}

static void ggml_compute_forward_group_norm(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_group_norm_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_mul_mat

#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool ggml_compute_forward_mul_mat_use_blas(const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                  struct ggml_tensor* dst) {
  // const int64_t ne00 = src0->ne[0];
  // const int64_t ne01 = src0->ne[1];

  const int64_t ne10 = src1->ne[0];

  const int64_t ne0 = dst->ne[0];
  const int64_t ne1 = dst->ne[1];

  // TODO: find the optimal values for these
  if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && (ne0 >= 32 && ne1 >= 32 && ne10 >= 32)) {
    /*printf("BLAS: %d %d %d %d %d\n", ne0, ne1, ne10, ne00, ne01);*/
    return true;
  }

  return false;
}
#endif

static void ggml_compute_forward_mul_mat(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  const enum ggml_type type = src0->type;

  const bool src1_cont = ggml_is_contiguous(src1);

  ggml_vec_dot_t const vec_dot = type_traits[type].vec_dot;
  enum ggml_type const vec_dot_type = type_traits[type].vec_dot_type;
  ggml_from_float_t const from_float_to_vec_dot = type_traits[vec_dot_type].from_float;

  GGML_ASSERT(ne0 == ne01);
  GGML_ASSERT(ne1 == ne11);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == type_traits[type].type_size);
  GGML_ASSERT(nb10 == sizeof(float));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  // broadcast factors
  const int64_t r2 = ne12 / ne02;
  const int64_t r3 = ne13 / ne03;

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

#if defined(GGML_USE_CLBLAST)
  if (ggml_cl_can_mul_mat(src0, src1, dst)) {
    // TODO: handle case when src0 is broadcast-able into src1 across 2nd,3rd dimension
    //       ref: https://github.com/ggerganov/ggml/pull/224
    GGML_ASSERT(ne02 == ne12);
    GGML_ASSERT(ne03 == ne13);

    if (params->ith == 0 && params->type == GGML_TASK_COMPUTE) {
      ggml_cl_mul_mat(src0, src1, dst, params->wdata, params->wsize);
    }
    return;
  }
#endif

#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
  if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
    if (params->ith != 0) {
      return;
    }

    if (params->type == GGML_TASK_INIT) {
      return;
    }

    if (params->type == GGML_TASK_FINALIZE) {
      return;
    }

    for (int64_t i13 = 0; i13 < ne13; i13++) {
      for (int64_t i12 = 0; i12 < ne12; i12++) {
        // broadcast src0 into src1 across 2nd,3rd dimension
        const int64_t i03 = i13 / r3;
        const int64_t i02 = i12 / r2;

        const void* x = (char*)src0->data + i02 * nb02 + i03 * nb03;
        const float* y = (float*)((char*)src1->data + i12 * nb12 + i13 * nb13);

        float* d = (float*)((char*)dst->data + i12 * nb2 + i13 * nb3);

        if (type != GGML_TYPE_F32) {
          float* const wdata = params->wdata;
          ggml_to_float_t const to_float = type_traits[type].to_float;

          size_t id = 0;
          for (int64_t i01 = 0; i01 < ne01; ++i01) {
            to_float((const char*)x + i01 * nb01, wdata + id, ne00);
            id += ne00;
          }

          assert(id * sizeof(float) <= params->wsize);
          x = wdata;
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ne11, ne01, ne10, 1.0f, y, ne10, x, ne00, 0.0f, d, ne01);
      }
    }

    // printf("CBLAS = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

    return;
  }
#endif

  if (params->type == GGML_TASK_INIT) {
    if (src1->type != vec_dot_type) {
      char* wdata = (char*)params->wdata;
      const size_t row_size = ne10 * type_traits[vec_dot_type].type_size / type_traits[vec_dot_type].blck_size;

      for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
          for (int64_t i11 = 0; i11 < ne11; ++i11) {
            from_float_to_vec_dot((float*)((char*)src1->data + i13 * nb13 + i12 * nb12 + i11 * nb11), (void*)wdata,
                                  ne10);
            wdata += row_size;
          }
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const void* wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
  const size_t row_size = ne10 * type_traits[vec_dot_type].type_size / type_traits[vec_dot_type].blck_size;

  const int64_t nr0 = ne01;                // src0 rows
  const int64_t nr1 = ne11 * ne12 * ne13;  // src1 rows

  // printf("nr0 = %lld, nr1 = %lld\n", nr0, nr1);

  // distribute the thread work across the inner or outer loop based on which one is larger

  const int64_t nth0 = nr0 > nr1 ? nth : 1;  // parallelize by src0 rows
  const int64_t nth1 = nr0 > nr1 ? 1 : nth;  // parallelize by src1 rows

  const int64_t ith0 = ith % nth0;
  const int64_t ith1 = ith / nth0;

  const int64_t dr0 = (nr0 + nth0 - 1) / nth0;
  const int64_t dr1 = (nr1 + nth1 - 1) / nth1;

  const int64_t ir010 = dr0 * ith0;
  const int64_t ir011 = MIN(ir010 + dr0, nr0);

  const int64_t ir110 = dr1 * ith1;
  const int64_t ir111 = MIN(ir110 + dr1, nr1);

  // printf("ir010 = %6lld, ir011 = %6lld, ir110 = %6lld, ir111 = %6lld\n", ir010, ir011, ir110, ir111);

  // threads with no work simply yield (not sure if it helps)
  if (ir010 >= ir011 || ir110 >= ir111) {
    sched_yield();
    return;
  }

  assert(ne12 % ne02 == 0);
  assert(ne13 % ne03 == 0);

  // block-tiling attempt
  const int64_t blck_0 = 16;
  const int64_t blck_1 = 16;

  // attempt to reduce false-sharing (does not seem to make a difference)
  float tmp[16];

  for (int64_t iir1 = ir110; iir1 < ir111; iir1 += blck_1) {
    for (int64_t iir0 = ir010; iir0 < ir011; iir0 += blck_0) {
      for (int64_t ir1 = iir1; ir1 < iir1 + blck_1 && ir1 < ir111; ++ir1) {
        const int64_t i13 = (ir1 / (ne12 * ne11));
        const int64_t i12 = (ir1 - i13 * ne12 * ne11) / ne11;
        const int64_t i11 = (ir1 - i13 * ne12 * ne11 - i12 * ne11);

        // broadcast src0 into src1
        const int64_t i03 = i13 / r3;
        const int64_t i02 = i12 / r2;

        const int64_t i1 = i11;
        const int64_t i2 = i12;
        const int64_t i3 = i13;

        const char* src0_row = (const char*)src0->data + (0 + i02 * nb02 + i03 * nb03);

        // desc: when src1 is not a contiguous memory block we have to calculate the offset using the strides
        //       if it is, then we have either copied the data to params->wdata and made it contiguous or we are using
        //       the original src1 data pointer, so we should index using the indices directly
        // TODO: this is a bit of a hack, we should probably have a better way to handle this
        const char* src1_col = (const char*)wdata + (src1_cont || src1->type != vec_dot_type
                                                         ? (i11 + i12 * ne11 + i13 * ne12 * ne11) * row_size
                                                         : (i11 * nb11 + i12 * nb12 + i13 * nb13));

        float* dst_col = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

        // for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
        //     vec_dot(ne00, &dst_col[ir0], src0_row + ir0*nb01, src1_col);
        // }

        for (int64_t ir0 = iir0; ir0 < iir0 + blck_0 && ir0 < ir011; ++ir0) {
          vec_dot(ne00, &tmp[ir0 - iir0], src0_row + ir0 * nb01, src1_col);
        }
        memcpy(&dst_col[iir0], tmp, (MIN(iir0 + blck_0, ir011) - iir0) * sizeof(float));
      }
    }
  }
}

// ggml_compute_forward_out_prod

static void ggml_compute_forward_out_prod_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  // int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  GGML_ASSERT(ne02 == ne12);
  GGML_ASSERT(ne03 == ne13);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  // we don't support permuted src0 or src1
  GGML_ASSERT(nb00 == sizeof(float));

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  // GGML_ASSERT(nb0 <= nb1);
  // GGML_ASSERT(nb1 <= nb2);
  // GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(ne0 == ne00);
  GGML_ASSERT(ne1 == ne10);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  // TODO: #if defined(GGML_USE_CUBLAS) ggml_cuda_out_prod
  // TODO: #if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CLBLAST)

  if (params->type == GGML_TASK_INIT) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float*)dst->data, 0);
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // dst[:,:,:,:] = 0
  // for i2,i3:
  //   for i1:
  //     for i01:
  //       for i0:
  //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

  // parallelize by last three dimensions

  // total rows in dst
  const int64_t nr = ne1 * ne2 * ne3;

  // rows per thread
  const int64_t dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int64_t ir0 = dr * ith;
  const int64_t ir1 = MIN(ir0 + dr, nr);

  // block-tiling attempt
  const int64_t blck_0 = MAX(GGML_VEC_MAD_UNROLL, 32);
  const int64_t blck_1 = 16;

  for (int64_t bir = ir0; bir < ir1; bir += blck_1) {
    const int64_t bir1 = MIN(bir + blck_1, ir1);
    for (int64_t bi01 = 0; bi01 < ne01; bi01 += blck_0) {
      const int64_t bne01 = MIN(bi01 + blck_0, ne01);
      for (int64_t ir = bir; ir < bir1; ++ir) {
        // dst indices
        const int64_t i3 = ir / (ne2 * ne1);
        const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
        const int64_t i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

        const int64_t i02 = i2;
        const int64_t i03 = i3;

        // const int64_t i10 = i1;
        const int64_t i12 = i2;
        const int64_t i13 = i3;

#if GGML_VEC_MAD_UNROLL > 2
        const int64_t bne01_unroll = bne01 - (bne01 % GGML_VEC_MAD_UNROLL);
        for (int64_t i01 = bi01; i01 < bne01_unroll; i01 += GGML_VEC_MAD_UNROLL) {
          const int64_t i11 = i01;

          float* s0 = (float*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
          float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
          float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

          ggml_vec_mad_f32_unroll(ne0, nb01, nb11, d, s0, s1);
        }
        for (int64_t i01 = bne01_unroll; i01 < bne01; ++i01) {
          const int64_t i11 = i01;

          float* s0 = (float*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
          float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
          float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

          ggml_vec_mad_f32(ne0, d, s0, *s1);
        }
#else
        for (int64_t i01 = bi01; i01 < bne01; ++i01) {
          const int64_t i11 = i01;

          float* s0 = (float*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
          float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
          float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

          ggml_vec_mad_f32(ne0, d, s0, *s1);
        }
#endif
      }
    }
  }

  // int64_t t1 = ggml_perf_time_us();
  // static int64_t acc = 0;
  // acc += t1 - t0;
  // if (t1 - t0 > 10) {
  //     printf("\n");
  //     printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
  //     printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
  //     printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);
  //     printf("nb10 = %5d, nb11 = %5d, nb12 = %5d, nb13 = %5d\n", nb10, nb11, nb12, nb13);

  //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1
  //    - t0), (int) acc);
  //}
}

static void ggml_compute_forward_out_prod_q_f32(const struct ggml_compute_params* params,
                                                const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                struct ggml_tensor* dst) {
  // int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS;

  const int ith = params->ith;
  const int nth = params->nth;

  const enum ggml_type type = src0->type;
  ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;

  GGML_ASSERT(ne02 == ne12);
  GGML_ASSERT(ne03 == ne13);
  GGML_ASSERT(ne2 == ne12);
  GGML_ASSERT(ne3 == ne13);

  // we don't support permuted src0 dim0
  GGML_ASSERT(nb00 == type_traits[type].type_size);

  // dst dim0 cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  // GGML_ASSERT(nb0 <= nb1);
  // GGML_ASSERT(nb1 <= nb2);
  // GGML_ASSERT(nb2 <= nb3);

  GGML_ASSERT(ne0 == ne00);
  GGML_ASSERT(ne1 == ne10);
  GGML_ASSERT(ne2 == ne02);
  GGML_ASSERT(ne3 == ne03);

  // nb01 >= nb00 - src0 is not transposed
  //   compute by src0 rows

  // TODO: #if defined(GGML_USE_CUBLAS) ggml_cuda_out_prod
  // TODO: #if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS) || defined(GGML_USE_CLBLAST)

  if (params->type == GGML_TASK_INIT) {
    ggml_vec_set_f32(ne0 * ne1 * ne2 * ne3, (float*)dst->data, 0);
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by last three dimensions

  // total rows in dst
  const int64_t nr = ne1 * ne2 * ne3;

  // rows per thread
  const int64_t dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int64_t ir0 = dr * ith;
  const int64_t ir1 = MIN(ir0 + dr, nr);

  // dst[:,:,:,:] = 0
  // for i2,i3:
  //   for i1:
  //     for i01:
  //       for i0:
  //         dst[i0,i1,i2,i3] += src0[i0,i01,i2,i3] * src1[i1,i01,i2,i3]

  float* wdata = (float*)params->wdata + (ne0 + CACHE_LINE_SIZE_F32) * ith;

  for (int64_t ir = ir0; ir < ir1; ++ir) {
    // dst indices
    const int64_t i3 = ir / (ne2 * ne1);
    const int64_t i2 = (ir - i3 * ne2 * ne1) / ne1;
    const int64_t i1 = (ir - i3 * ne2 * ne1 - i2 * ne1);

    const int64_t i02 = i2;
    const int64_t i03 = i3;

    // const int64_t i10 = i1;
    const int64_t i12 = i2;
    const int64_t i13 = i3;

    for (int64_t i01 = 0; i01 < ne01; ++i01) {
      const int64_t i11 = i01;

      float* s0 = (float*)((char*)src0->data + (i01 * nb01 + i02 * nb02 + i03 * nb03));
      float* s1 = (float*)((char*)src1->data + (i1 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13));
      float* d = (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3));

      dequantize_row_q(s0, wdata, ne0);
      ggml_vec_mad_f32(ne0, d, wdata, *s1);
    }
  }

  // int64_t t1 = ggml_perf_time_us();
  // static int64_t acc = 0;
  // acc += t1 - t0;
  // if (t1 - t0 > 10) {
  //     printf("\n");
  //     printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
  //     printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
  //     printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);
  //     printf("nb10 = %5d, nb11 = %5d, nb12 = %5d, nb13 = %5d\n", nb10, nb11, nb12, nb13);

  //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1
  //    - t0), (int) acc);
  //}
}

static void ggml_compute_forward_out_prod(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: {
      ggml_compute_forward_out_prod_q_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16: {
      GGML_ASSERT(false);  // todo
                           // ggml_compute_forward_out_prod_f16_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_out_prod_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_scale

static void ggml_compute_forward_scale_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                           const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous(src0));
  GGML_ASSERT(ggml_is_contiguous(dst));
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_scalar(src1));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // scale factor
  const float v = *(float*)src1->data;

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const size_t nb01 = src0->nb[1];

  const size_t nb1 = dst->nb[1];

  for (int i1 = ir0; i1 < ir1; i1++) {
    if (dst->data != src0->data) {
      // src0 is same shape as dst => same indices
      memcpy((char*)dst->data + i1 * nb1, (char*)src0->data + i1 * nb01, nc * sizeof(float));
    }
    ggml_vec_scale_f32(nc, (float*)((char*)dst->data + i1 * nb1), v);
  }
}

static void ggml_compute_forward_scale(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                       const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_scale_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_set

static void ggml_compute_forward_set_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

  // view src0 and dst with these strides and data offset inbytes during set
  // nb0 is implicitely element_size because src0 and dst are contiguous
  size_t nb1 = ((int32_t*)dst->op_params)[0];
  size_t nb2 = ((int32_t*)dst->op_params)[1];
  size_t nb3 = ((int32_t*)dst->op_params)[2];
  size_t offset = ((int32_t*)dst->op_params)[3];
  bool inplace = (bool)((int32_t*)dst->op_params)[4];

  if (!inplace && (params->type == GGML_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase
    memcpy(((char*)dst->data), ((char*)src0->data), ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(src1);
  const int nc = src1->ne[0];

  GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne)
  GGML_TENSOR_LOCALS(size_t, nb1, src1, nb)

  // src0 and dst as viewed during set
  const size_t nb0 = type_traits[src0->type].type_size;

  const int im0 = (ne10 == 0 ? 0 : ne10 - 1);
  const int im1 = (ne11 == 0 ? 0 : ne11 - 1);
  const int im2 = (ne12 == 0 ? 0 : ne12 - 1);
  const int im3 = (ne13 == 0 ? 0 : ne13 - 1);

  GGML_ASSERT(offset + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(dst));

  GGML_ASSERT(nb10 == sizeof(float));

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // src0 and dst are viewed with shape of src1 and offset
    // => same indices
    const int i3 = ir / (ne12 * ne11);
    const int i2 = (ir - i3 * ne12 * ne11) / ne11;
    const int i1 = (ir - i3 * ne12 * ne11 - i2 * ne11);

    ggml_vec_cpy_f32(nc, (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + offset),
                     (float*)((char*)src1->data + i3 * nb13 + i2 * nb12 + i1 * nb11));
  }
}

static void ggml_compute_forward_set(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_set_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_cpy

static void ggml_compute_forward_cpy(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                     struct ggml_tensor* dst) {
  ggml_compute_forward_dup(params, src0, dst);
}

// ggml_compute_forward_cont

static void ggml_compute_forward_cont(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  ggml_compute_forward_dup(params, src0, dst);
}

// ggml_compute_forward_reshape

static void ggml_compute_forward_reshape(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  // NOP
  // UNUSED(params);
  // UNUSED(src0);
  // UNUSED(dst);
}

// ggml_compute_forward_view

static void ggml_compute_forward_view(const struct ggml_compute_params* params, const struct ggml_tensor* src0) {
  // NOP
  // UNUSED(params);
  // UNUSED(src0);
}

// ggml_compute_forward_permute

static void ggml_compute_forward_permute(const struct ggml_compute_params* params, const struct ggml_tensor* src0) {
  // NOP
  // UNUSED(params);
  // UNUSED(src0);
}

// ggml_compute_forward_transpose

static void ggml_compute_forward_transpose(const struct ggml_compute_params* params, const struct ggml_tensor* src0) {
  // NOP
  // UNUSED(params);
  // UNUSED(src0);
}

// ggml_compute_forward_get_rows

static void ggml_compute_forward_get_rows_q(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);
  const enum ggml_type type = src0->type;
  ggml_to_float_t const dequantize_row_q = type_traits[type].to_float;

  assert(dst->ne[0] == nc);
  assert(dst->ne[1] == nr);
  assert(src0->nb[0] == type_traits[type].type_size);

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    dequantize_row_q((const void*)((char*)src0->data + r * src0->nb[1]), (float*)((char*)dst->data + i * dst->nb[1]),
                     nc);
  }
}

static void ggml_compute_forward_get_rows_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);

  assert(dst->ne[0] == nc);
  assert(dst->ne[1] == nr);
  assert(src0->nb[0] == sizeof(ggml_fp16_t));

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    for (int j = 0; j < nc; ++j) {
      ggml_fp16_t v = ((ggml_fp16_t*)((char*)src0->data + r * src0->nb[1]))[j];
      ((float*)((char*)dst->data + i * dst->nb[1]))[j] = GGML_FP16_TO_FP32(v);
    }
  }
}

static void ggml_compute_forward_get_rows_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);

  assert(dst->ne[0] == nc);
  assert(dst->ne[1] == nr);
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    ggml_vec_cpy_f32(nc, (float*)((char*)dst->data + i * dst->nb[1]), (float*)((char*)src0->data + r * src0->nb[1]));
  }
}

static void ggml_compute_forward_get_rows(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K: {
      ggml_compute_forward_get_rows_q(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F16: {
      ggml_compute_forward_get_rows_f16(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_get_rows_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }

  // static bool first = true;
  // printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
  // if (first) {
  //     first = false;
  // } else {
  //     for (int k = 0; k < dst->ne[1]; ++k) {
  //         for (int j = 0; j < dst->ne[0]/16; ++j) {
  //             for (int i = 0; i < 16; ++i) {
  //                 printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
  //             }
  //             printf("\n");
  //         }
  //         printf("\n");
  //     }
  //     printf("\n");
  //     exit(0);
  // }
}

// ggml_compute_forward_get_rows_back

static void ggml_compute_forward_get_rows_back_f32_f16(const struct ggml_compute_params* params,
                                                       const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                       struct ggml_tensor* dst) {
  GGML_ASSERT(params->ith == 0);
  GGML_ASSERT(ggml_is_contiguous(dst));

  // ggml_compute_forward_dup_same_cont(params, opt0, dst);

  if (params->type == GGML_TASK_INIT) {
    memset(dst->data, 0, ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);

  GGML_ASSERT(dst->ne[0] == nc);
  GGML_ASSERT(src0->nb[0] == sizeof(ggml_fp16_t));

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    for (int j = 0; j < nc; ++j) {
      ggml_fp16_t v = ((ggml_fp16_t*)((char*)src0->data + i * src0->nb[1]))[j];
      ((float*)((char*)dst->data + r * dst->nb[1]))[j] += GGML_FP16_TO_FP32(v);
    }
  }
}

static void ggml_compute_forward_get_rows_back_f32(const struct ggml_compute_params* params,
                                                   const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                   struct ggml_tensor* dst) {
  GGML_ASSERT(params->ith == 0);
  GGML_ASSERT(ggml_is_contiguous(dst));

  // ggml_compute_forward_dup_same_cont(params, opt0, dst);

  if (params->type == GGML_TASK_INIT) {
    memset(dst->data, 0, ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int nc = src0->ne[0];
  const int nr = ggml_nelements(src1);

  GGML_ASSERT(dst->ne[0] == nc);
  GGML_ASSERT(src0->nb[0] == sizeof(float));

  for (int i = 0; i < nr; ++i) {
    const int r = ((int32_t*)src1->data)[i];

    ggml_vec_add_f32(nc, (float*)((char*)dst->data + r * dst->nb[1]), (float*)((char*)dst->data + r * dst->nb[1]),
                     (float*)((char*)src0->data + i * src0->nb[1]));
  }
}

static void ggml_compute_forward_get_rows_back(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_get_rows_back_f32_f16(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_get_rows_back_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }

  // static bool first = true;
  // printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
  // if (first) {
  //     first = false;
  // } else {
  //     for (int k = 0; k < dst->ne[1]; ++k) {
  //         for (int j = 0; j < dst->ne[0]/16; ++j) {
  //             for (int i = 0; i < 16; ++i) {
  //                 printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
  //             }
  //             printf("\n");
  //         }
  //         printf("\n");
  //     }
  //     printf("\n");
  //     exit(0);
  // }
}

// ggml_compute_forward_diag

static void ggml_compute_forward_diag_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  GGML_ASSERT(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  GGML_TENSOR_UNARY_OP_LOCALS

  GGML_ASSERT(ne00 == ne0);
  GGML_ASSERT(ne00 == ne1);
  GGML_ASSERT(ne01 == 1);
  GGML_ASSERT(ne02 == ne2);
  GGML_ASSERT(ne03 == ne3);

  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb0 == sizeof(float));

  for (int i3 = 0; i3 < ne3; i3++) {
    for (int i2 = 0; i2 < ne2; i2++) {
      for (int i1 = 0; i1 < ne1; i1++) {
        float* d = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1);
        float* s = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02);
        for (int i0 = 0; i0 < i1; i0++) {
          d[i0] = 0;
        }
        d[i1] = s[i1];
        for (int i0 = i1 + 1; i0 < ne0; i0++) {
          d[i0] = 0;
        }
      }
    }
  }
}

static void ggml_compute_forward_diag(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_diag_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_diag_mask_inf

static void ggml_compute_forward_diag_mask_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               struct ggml_tensor* dst, const float value) {
  const int ith = params->ith;
  const int nth = params->nth;

  const int n_past = ((int32_t*)dst->op_params)[0];
  const bool inplace = src0->data == dst->data;

  GGML_ASSERT(n_past >= 0);

  if (!inplace && (params->type == GGML_TASK_INIT)) {
    // memcpy needs to be synchronized across threads to avoid race conditions.
    // => do it in INIT phase
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    memcpy(((char*)dst->data), ((char*)src0->data), ggml_nbytes(dst));
  }

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];
  const int nr = src0->ne[1];
  const int nz = n / nr;

  GGML_ASSERT(dst->nb[0] == sizeof(float));
  GGML_ASSERT(src0->nb[0] == sizeof(float));

  for (int k = 0; k < nz; k++) {
    for (int j = ith; j < nr; j += nth) {
      for (int i = n_past; i < nc; i++) {
        if (i > n_past + j) {
          *(float*)((char*)dst->data + k * dst->nb[2] + j * dst->nb[1] + i * dst->nb[0]) = value;
        }
      }
    }
  }
}

static void ggml_compute_forward_diag_mask_inf(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_diag_mask_f32(params, src0, dst, -INFINITY);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

static void ggml_compute_forward_diag_mask_zero(const struct ggml_compute_params* params,
                                                const struct ggml_tensor* src0, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_diag_mask_f32(params, src0, dst, 0);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_soft_max

static void ggml_compute_forward_soft_max_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous(src0));
  GGML_ASSERT(ggml_is_contiguous(dst));
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* sp = (float*)((char*)src0->data + i1 * src0->nb[1]);
    float* dp = (float*)((char*)dst->data + i1 * dst->nb[1]);

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      // printf("p[%d] = %f\n", i, p[i]);
      assert(!isnan(sp[i]));
    }
#endif

    float max = -INFINITY;
    ggml_vec_max_f32(nc, &max, sp);

    ggml_float sum = 0.0;

    uint16_t scvt;
    for (int i = 0; i < nc; i++) {
      if (sp[i] == -INFINITY) {
        dp[i] = 0.0f;
      } else {
        // const float val = (sp[i] == -INFINITY) ? 0.0 : exp(sp[i] - max);
        ggml_fp16_t s = GGML_FP32_TO_FP16(sp[i] - max);
        memcpy(&scvt, &s, sizeof(scvt));
        const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt]);
        sum += (ggml_float)val;
        dp[i] = val;
      }
    }

    assert(sum > 0.0);

    sum = 1.0 / sum;
    ggml_vec_scale_f32(nc, dp, sum);

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      assert(!isnan(dp[i]));
      assert(!isinf(dp[i]));
    }
#endif
  }
}

static void ggml_compute_forward_soft_max(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_soft_max_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_soft_max_back

static void ggml_compute_forward_soft_max_back_f32(const struct ggml_compute_params* params,
                                                   const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                   struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous(src0));
  GGML_ASSERT(ggml_is_contiguous(src1));
  GGML_ASSERT(ggml_is_contiguous(dst));
  GGML_ASSERT(ggml_are_same_shape(src0, dst));
  GGML_ASSERT(ggml_are_same_shape(src1, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // TODO: handle transposed/permuted matrices

  const int ith = params->ith;
  const int nth = params->nth;

  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dy = (float*)((char*)src0->data + i1 * src0->nb[1]);
    float* y = (float*)((char*)src1->data + i1 * src1->nb[1]);
    float* dx = (float*)((char*)dst->data + i1 * dst->nb[1]);

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      // printf("p[%d] = %f\n", i, p[i]);
      assert(!isnan(dy[i]));
      assert(!isnan(y[i]));
    }
#endif
    // Jii = yi - yi*yi
    // Jij = -yi*yj
    // J = diag(y)-y.T*y
    // dx = J * dy
    // dxk = sum_i(Jki * dyi)
    // dxk = sum_i(-yk*yi * dyi) - (-yk*yk)*dyk + (yk - yk*yk)*dyk
    // dxk = sum_i(-yk*yi * dyi) + yk*yk*dyk + yk*dyk - yk*yk*dyk
    // dxk = sum_i(-yk*yi * dyi) + yk*dyk
    // dxk = -yk * sum_i(yi * dyi) + yk*dyk
    // dxk = -yk * dot(y, dy) + yk*dyk
    // dxk = yk * (- dot(y, dy) + dyk)
    // dxk = yk * (dyk - dot(y, dy))
    //
    // post-order:
    // dot_y_dy := dot(y, dy)
    // dx := dy
    // dx := dx - dot_y_dy
    // dx := dx * y

    // linear runtime, no additional memory
    float dot_y_dy = 0;
    ggml_vec_dot_f32(nc, &dot_y_dy, y, dy);
    ggml_vec_cpy_f32(nc, dx, dy);
    ggml_vec_acc1_f32(nc, dx, -dot_y_dy);
    ggml_vec_mul_f32(nc, dx, dx, y);

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      assert(!isnan(dx[i]));
      assert(!isinf(dx[i]));
    }
#endif
  }
}

static void ggml_compute_forward_soft_max_back(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_soft_max_back_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_alibi

static void ggml_compute_forward_alibi_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                           struct ggml_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n_past = ((int32_t*)dst->op_params)[0];
  const int n_head = ((int32_t*)dst->op_params)[1];
  float max_bias;
  memcpy(&max_bias, (int32_t*)dst->op_params + 2, sizeof(float));

  assert(n_past >= 0);

  const int ne0 = src0->ne[0];  // all_seq_len = n_past + ne1
  const int ne1 = src0->ne[1];  // seq_len_without_past
  const int ne2 = src0->ne[2];  // n_head -> this is k
  // const int ne3 = src0->ne[3]; // 1 -> bsz

  const int n = ggml_nrows(src0);
  const int ne2_ne3 = n / ne1;  // ne2*ne3

  const int nb0 = src0->nb[0];
  const int nb1 = src0->nb[1];
  const int nb2 = src0->nb[2];
  // const int nb3 = src0->nb[3];

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(ne1 + n_past == ne0);
  GGML_ASSERT(n_head == ne2);

  // add alibi to src0 (KQ_scaled)
  const int n_heads_log2_floor = 1 << (int)floor(log2(n_head));

  const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

  for (int i = 0; i < ne0; i++) {
    for (int j = 0; j < ne1; j++) {
      for (int k = 0; k < ne2_ne3; k++) {
        float* const src = (float*)((char*)src0->data + i * nb0 + j * nb1 + k * nb2);
        float* pdst = (float*)((char*)dst->data + i * nb0 + j * nb1 + k * nb2);

        // TODO: k*nb2 or k*nb3

        float m_k;

        if (k < n_heads_log2_floor) {
          m_k = powf(m0, k + 1);
        } else {
          m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
        }

        pdst[0] = i * m_k + src[0];
      }
    }
  }
}

static void ggml_compute_forward_alibi_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                           struct ggml_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // const int n_past = ((int32_t *) dst->op_params)[0];
  const int n_head = ((int32_t*)dst->op_params)[1];
  float max_bias;
  memcpy(&max_bias, (int32_t*)dst->op_params + 2, sizeof(float));

  const int ne0 = src0->ne[0];  // all_seq_len = n_past + ne1
  const int ne1 = src0->ne[1];  // seq_len_without_past
  const int ne2 = src0->ne[2];  // n_head -> this is k
  // const int ne3 = src0->ne[3]; // 1 -> bsz

  const int n = ggml_nrows(src0);
  const int ne2_ne3 = n / ne1;  // ne2*ne3

  const int nb0 = src0->nb[0];
  const int nb1 = src0->nb[1];
  const int nb2 = src0->nb[2];
  // const int nb3 = src0->nb[3];

  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));
  // GGML_ASSERT(ne1 + n_past == ne0); (void) n_past;
  GGML_ASSERT(n_head == ne2);

  // add alibi to src0 (KQ_scaled)
  const int n_heads_log2_floor = 1 << (int)floor(log2(n_head));

  const float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
  const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_heads_log2_floor);

  for (int i = 0; i < ne0; i++) {
    for (int j = 0; j < ne1; j++) {
      for (int k = 0; k < ne2_ne3; k++) {
        ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i * nb0 + j * nb1 + k * nb2);
        float* pdst = (float*)((char*)dst->data + i * nb0 + j * nb1 + k * nb2);

        // TODO: k*nb2 or k*nb3

        float m_k;

        if (k < n_heads_log2_floor) {
          m_k = powf(m0, k + 1);
        } else {
          m_k = powf(m1, 2 * (k - n_heads_log2_floor) + 1);
        }

        // we return F32
        pdst[0] = i * m_k + GGML_FP16_TO_FP32(src[0]);
      }
    }
  }
}

static void ggml_compute_forward_alibi(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                       struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_alibi_f16(params, src0, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_alibi_f32(params, src0, dst);
    } break;
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_K:
    case GGML_TYPE_I8:
    case GGML_TYPE_I16:
    case GGML_TYPE_I32:
    case GGML_TYPE_COUNT: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_clamp

static void ggml_compute_forward_clamp_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                           struct ggml_tensor* dst) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  float min;
  float max;
  memcpy(&min, (float*)dst->op_params + 0, sizeof(float));
  memcpy(&max, (float*)dst->op_params + 1, sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  const size_t nb00 = src0->nb[0];
  const size_t nb01 = src0->nb[1];

  const size_t nb0 = dst->nb[0];
  const size_t nb1 = dst->nb[1];

  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb00 == sizeof(float));

  for (int j = ith; j < n; j += nth) {
    float* dst_ptr = (float*)((char*)dst->data + j * nb1);
    float* src0_ptr = (float*)((char*)src0->data + j * nb01);

    for (int i = 0; i < nc; i++) {
      dst_ptr[i] = MAX(MIN(src0_ptr[i], max), min);
    }
  }
}

static void ggml_compute_forward_clamp(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                       struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_clamp_f32(params, src0, dst);
    } break;
    case GGML_TYPE_F16:
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q5_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q8_1:
    case GGML_TYPE_Q2_K:
    case GGML_TYPE_Q3_K:
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q8_K:
    case GGML_TYPE_I8:
    case GGML_TYPE_I16:
    case GGML_TYPE_I32:
    case GGML_TYPE_COUNT: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_rope

static void ggml_compute_forward_rope_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  float freq_base;
  float freq_scale;

  // these two only relevant for xPos RoPE:
  float xpos_base;
  bool xpos_down;

  // const int n_past = ((int32_t *) dst->op_params)[0];
  const int n_dims = ((int32_t*)dst->op_params)[1];
  const int mode = ((int32_t*)dst->op_params)[2];
  const int n_ctx = ((int32_t*)dst->op_params)[3];
  memcpy(&freq_base, (int32_t*)dst->op_params + 4, sizeof(float));
  memcpy(&freq_scale, (int32_t*)dst->op_params + 5, sizeof(float));
  memcpy(&xpos_base, (int32_t*)dst->op_params + 6, sizeof(float));
  memcpy(&xpos_down, (int32_t*)dst->op_params + 7, sizeof(bool));

  GGML_TENSOR_UNARY_OP_LOCALS

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  GGML_ASSERT(nb00 == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(dst);

  GGML_ASSERT(n_dims <= ne0);
  GGML_ASSERT(n_dims % 2 == 0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(freq_base, -2.0f / n_dims);

  const bool is_neox = mode & 2;
  const bool is_glm = mode & 4;

  const int32_t* pos = (const int32_t*)src1->data;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = 0; i2 < ne2; i2++) {
      const int64_t p = pos[i2];
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) continue;
        if (ir > ir1) break;

        float theta = freq_scale * (float)p;

        if (is_glm) {
          theta = MIN(p, n_ctx - 2);
          float block_theta = MAX(p - (n_ctx - 2), 0);
          for (int64_t i0 = 0; i0 < ne0 / 4; i0++) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);
            const float cos_block_theta = cosf(block_theta);
            const float sin_block_theta = sinf(block_theta);

            theta *= theta_scale;
            block_theta *= theta_scale;

            const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims / 2];
            const float x2 = src[n_dims];
            const float x3 = src[n_dims / 2 * 3];

            dst_data[0] = x0 * cos_theta - x1 * sin_theta;
            dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            dst_data[n_dims] = x2 * cos_block_theta - x3 * sin_block_theta;
            dst_data[n_dims / 2 * 3] = x2 * sin_block_theta + x3 * cos_block_theta;
          }
        } else if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);
            // zeta scaling for xPos only:
            float zeta = xpos_base != 0.0f ? powf((i0 + 0.4f * ne0) / (1.4f * ne0), p / xpos_base) : 1.0f;
            if (xpos_down) zeta = 1.0f / zeta;

            theta *= theta_scale;

            const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = src[0];
            const float x1 = src[1];

            dst_data[0] = x0 * cos_theta * zeta - x1 * sin_theta * zeta;
            dst_data[1] = x0 * sin_theta * zeta + x1 * cos_theta * zeta;
          }
        } else {
          // TODO: this might be wrong for ne0 != n_dims - need double check
          // ref:
          // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#LL251C1-L294C28
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const float* const src = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float x0 = src[0];
              const float x1 = src[n_dims / 2];

              dst_data[0] = x0 * cos_theta - x1 * sin_theta;
              dst_data[n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_rope_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  float freq_base;
  float freq_scale;

  // const int n_past = ((int32_t *) dst->op_params)[0];
  const int n_dims = ((int32_t*)dst->op_params)[1];
  const int mode = ((int32_t*)dst->op_params)[2];
  const int n_ctx = ((int32_t*)dst->op_params)[3];
  memcpy(&freq_base, (int32_t*)dst->op_params + 4, sizeof(float));
  memcpy(&freq_scale, (int32_t*)dst->op_params + 5, sizeof(float));

  GGML_TENSOR_UNARY_OP_LOCALS

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  GGML_ASSERT(nb0 == sizeof(ggml_fp16_t));

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(dst);

  GGML_ASSERT(n_dims <= ne0);
  GGML_ASSERT(n_dims % 2 == 0);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(freq_base, -2.0f / n_dims);

  const bool is_neox = mode & 2;
  const bool is_glm = mode & 4;

  const int32_t* pos = (const int32_t*)src1->data;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = 0; i2 < ne2; i2++) {
      const int64_t p = pos[i2];
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) continue;
        if (ir > ir1) break;

        float theta = freq_scale * (float)p;

        if (is_glm) {
          theta = MIN(p, n_ctx - 2);
          float block_theta = MAX(p - (n_ctx - 2), 0);
          for (int64_t i0 = 0; i0 < ne0 / 4; i0++) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);
            const float cos_block_theta = cosf(block_theta);
            const float sin_block_theta = sinf(block_theta);

            theta *= theta_scale;
            block_theta *= theta_scale;

            const ggml_fp16_t* const src =
                (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            ggml_fp16_t* dst_data = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = GGML_FP16_TO_FP32(src[0]);
            const float x1 = GGML_FP16_TO_FP32(src[n_dims / 2]);
            const float x2 = GGML_FP16_TO_FP32(src[n_dims]);
            const float x3 = GGML_FP16_TO_FP32(src[n_dims / 2 * 3]);

            dst_data[0] = GGML_FP32_TO_FP16(x0 * cos_theta - x1 * sin_theta);
            dst_data[n_dims / 2] = GGML_FP32_TO_FP16(x0 * sin_theta + x1 * cos_theta);
            dst_data[n_dims] = GGML_FP32_TO_FP16(x2 * cos_block_theta - x3 * sin_block_theta);
            dst_data[n_dims / 2 * 3] = GGML_FP32_TO_FP16(x2 * sin_block_theta + x3 * cos_block_theta);
          }
        }
        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const ggml_fp16_t* const src =
                (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            ggml_fp16_t* dst_data = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float x0 = GGML_FP16_TO_FP32(src[0]);
            const float x1 = GGML_FP16_TO_FP32(src[1]);

            dst_data[0] = GGML_FP32_TO_FP16(x0 * cos_theta - x1 * sin_theta);
            dst_data[1] = GGML_FP32_TO_FP16(x0 * sin_theta + x1 * cos_theta);
          }
        } else {
          // TODO: this might be wrong for ne0 != n_dims - need double check
          // ref:
          // https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neox/modeling_gpt_neox.py#LL251C1-L294C28
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const ggml_fp16_t* const src =
                  (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              ggml_fp16_t* dst_data = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float x0 = GGML_FP16_TO_FP32(src[0]);
              const float x1 = GGML_FP16_TO_FP32(src[n_dims / 2]);

              dst_data[0] = GGML_FP32_TO_FP16(x0 * cos_theta - x1 * sin_theta);
              dst_data[n_dims / 2] = GGML_FP32_TO_FP16(x0 * sin_theta + x1 * cos_theta);
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_rope(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                      const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_rope_f16(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_rope_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_rope_back

static void ggml_compute_forward_rope_back_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // y = rope(x, src1)
  // dx = rope_back(dy, src1)
  // src0 is dy, src1 contains options

  float freq_base;
  float freq_scale;

  // these two only relevant for xPos RoPE:
  float xpos_base;
  bool xpos_down;

  // const int n_past = ((int32_t *) dst->op_params)[0];
  const int n_dims = ((int32_t*)dst->op_params)[1];
  const int mode = ((int32_t*)dst->op_params)[2];
  const int n_ctx = ((int32_t*)dst->op_params)[3];
  // UNUSED(n_ctx);
  memcpy(&freq_base, (int32_t*)dst->op_params + 4, sizeof(float));
  memcpy(&freq_scale, (int32_t*)dst->op_params + 5, sizeof(float));
  memcpy(&xpos_base, (int32_t*)dst->op_params + 6, sizeof(float));
  memcpy(&xpos_down, (int32_t*)dst->op_params + 7, sizeof(bool));

  GGML_TENSOR_UNARY_OP_LOCALS

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  assert(nb0 == sizeof(float));

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(dst);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(freq_base, -2.0f / n_dims);

  const bool is_neox = mode & 2;

  const int32_t* pos = (const int32_t*)src1->data;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = 0; i2 < ne2; i2++) {
      const int64_t p = pos[i2];
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) continue;
        if (ir > ir1) break;

        float theta = freq_scale * (float)p;

        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);
            // zeta scaling for xPos only:
            float zeta = xpos_base != 0.0f ? powf((i0 + 0.4f * ne0) / (1.4f * ne0), p / xpos_base) : 1.0f;
            if (xpos_down) zeta = 1.0f / zeta;

            theta *= theta_scale;

            const float* const dy = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            float* dx = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float dy0 = dy[0];
            const float dy1 = dy[1];

            dx[0] = dy0 * cos_theta * zeta + dy1 * sin_theta * zeta;
            dx[1] = -dy0 * sin_theta * zeta + dy1 * cos_theta * zeta;
          }
        } else {
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const float* const dy = (float*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              float* dx = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float dy0 = dy[0];
              const float dy1 = dy[n_dims / 2];

              dx[0] = dy0 * cos_theta + dy1 * sin_theta;
              dx[n_dims / 2] = -dy0 * sin_theta + dy1 * cos_theta;
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_rope_back_f16(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // y = rope(x, src1)
  // dx = rope_back(dy, src1)
  // src0 is dy, src1 contains options

  // const int n_past = ((int32_t *) dst->op_params)[0];
  const int n_dims = ((int32_t*)dst->op_params)[1];
  const int mode = ((int32_t*)dst->op_params)[2];

  GGML_TENSOR_UNARY_OP_LOCALS

  // printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
  // printf("n_past = %d, ne2 = %d\n", n_past, ne2);

  assert(nb0 == sizeof(ggml_fp16_t));

  const int ith = params->ith;
  const int nth = params->nth;

  const int nr = ggml_nrows(dst);

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  // row index used to determine which thread to use
  int ir = 0;

  const float theta_scale = powf(10000.0, -2.0f / n_dims);

  const bool is_neox = mode & 2;

  const int32_t* pos = (const int32_t*)src1->data;

  for (int64_t i3 = 0; i3 < ne3; i3++) {
    for (int64_t i2 = 0; i2 < ne2; i2++) {
      const int64_t p = pos[i2];
      for (int64_t i1 = 0; i1 < ne1; i1++) {
        if (ir++ < ir0) continue;
        if (ir > ir1) break;

        float theta = (float)p;

        if (!is_neox) {
          for (int64_t i0 = 0; i0 < ne0; i0 += 2) {
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            theta *= theta_scale;

            const ggml_fp16_t* const dy =
                (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
            ggml_fp16_t* dx = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

            const float dy0 = GGML_FP16_TO_FP32(dy[0]);
            const float dy1 = GGML_FP16_TO_FP32(dy[1]);

            dx[0] = GGML_FP32_TO_FP16(dy0 * cos_theta + dy1 * sin_theta);
            dx[1] = GGML_FP32_TO_FP16(-dy0 * sin_theta + dy1 * cos_theta);
          }
        } else {
          for (int64_t ib = 0; ib < ne0 / n_dims; ++ib) {
            for (int64_t ic = 0; ic < n_dims; ic += 2) {
              const float cos_theta = cosf(theta);
              const float sin_theta = sinf(theta);

              theta *= theta_scale;

              const int64_t i0 = ib * n_dims + ic / 2;

              const ggml_fp16_t* const dy =
                  (ggml_fp16_t*)((char*)src0->data + i3 * nb03 + i2 * nb02 + i1 * nb01 + i0 * nb00);
              ggml_fp16_t* dx = (ggml_fp16_t*)((char*)dst->data + i3 * nb3 + i2 * nb2 + i1 * nb1 + i0 * nb0);

              const float dy0 = GGML_FP16_TO_FP32(dy[0]);
              const float dy1 = GGML_FP16_TO_FP32(dy[n_dims / 2]);

              dx[0] = GGML_FP32_TO_FP16(dy0 * cos_theta + dy1 * sin_theta);
              dx[n_dims / 2] = GGML_FP32_TO_FP16(-dy0 * sin_theta + dy1 * cos_theta);
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_rope_back(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                           const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_rope_back_f16(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_rope_back_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_conv_1d

static void ggml_compute_forward_conv_1d_s1_ph_f16_f32(const struct ggml_compute_params* params,
                                                       const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                       struct ggml_tensor* dst) {
  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ggml_up32(ne01);

  GGML_ASSERT(ne00 % 2 == 1);  // TODO: support even kernel sizes
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb10 == sizeof(float));

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i02 * nb02 + i01 * nb01);
          ggml_fp16_t* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = (float*)((char*)src1->data + i11 * nb11);
        ggml_fp16_t* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = GGML_FP32_TO_FP16(src[i10]);
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = (float*)((char*)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ggml_vec_dot_f16(ew0, &v, (ggml_fp16_t*)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                         (ggml_fp16_t*)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

static void ggml_compute_forward_conv_1d_s1_ph_f32(const struct ggml_compute_params* params,
                                                   const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                   struct ggml_tensor* dst) {
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ggml_up32(ne01);

  GGML_ASSERT(ne00 % 2 == 1);  // TODO: support even kernel sizes
  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float* const wdata = (float*)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* const src = (float*)((char*)src0->data + i02 * nb02 + i01 * nb01);
          float* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float* const wdata = (float*)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = (float*)((char*)src1->data + i11 * nb11);
        float* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = (float*)((char*)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; ++i0) {
      dst_data[i0] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ggml_vec_dot_f32(ew0, &v, (float*)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                         (float*)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0] += v;
      }
    }
  }
}

static void ggml_compute_forward_conv_1d_s1_ph(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_conv_1d_s1_ph_f16_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_conv_1d_s1_ph_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

static void ggml_compute_forward_conv_1d_s2_ph_f16_f32(const struct ggml_compute_params* params,
                                                       const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                       struct ggml_tensor* dst) {
  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ggml_up32(ne01);

  GGML_ASSERT(ne00 % 2 == 1);  // TODO: support even kernel sizes
  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb10 == sizeof(float));

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i02 * nb02 + i01 * nb01);
          ggml_fp16_t* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = (float*)((char*)src1->data + i11 * nb11);
        ggml_fp16_t* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = GGML_FP32_TO_FP16(src[i10]);
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = (float*)((char*)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ggml_vec_dot_f16(ew0, &v, (ggml_fp16_t*)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                         (ggml_fp16_t*)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

static void ggml_compute_forward_conv_1d_s2_ph_f32(const struct ggml_compute_params* params,
                                                   const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                   struct ggml_tensor* dst) {
  GGML_ASSERT(src0->type == GGML_TYPE_F32);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00;
  const int nh = nk / 2;

  const int ew0 = ggml_up32(ne01);

  GGML_ASSERT(ne00 % 2 == 1);  // TODO: support even kernel sizes
  GGML_ASSERT(nb00 == sizeof(float));
  GGML_ASSERT(nb10 == sizeof(float));

  if (params->type == GGML_TASK_INIT) {
    // TODO: fix this memset (wsize is overestimated)
    memset(params->wdata, 0, params->wsize);

    // prepare kernel data (src0)
    {
      float* const wdata = (float*)params->wdata + 0;

      for (int64_t i02 = 0; i02 < ne02; i02++) {
        for (int64_t i01 = 0; i01 < ne01; i01++) {
          const float* const src = (float*)((char*)src0->data + i02 * nb02 + i01 * nb01);
          float* dst_data = wdata + i02 * ew0 * ne00;
          for (int64_t i00 = 0; i00 < ne00; i00++) {
            dst_data[i00 * ew0 + i01] = src[i00];
          }
        }
      }
    }

    // prepare source data (src1)
    {
      float* const wdata = (float*)params->wdata + ne02 * ew0 * ne00;

      for (int64_t i11 = 0; i11 < ne11; i11++) {
        const float* const src = (float*)((char*)src1->data + i11 * nb11);
        float* dst_data = wdata;
        for (int64_t i10 = 0; i10 < ne10; i10++) {
          dst_data[(i10 + nh) * ew0 + i11] = src[i10];
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total rows in dst
  const int nr = ne02;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* dst_data = (float*)((char*)dst->data + i1 * nb1);
    for (int64_t i0 = 0; i0 < ne10; i0 += 2) {
      dst_data[i0 / 2] = 0;
      for (int k = -nh; k <= nh; k++) {
        float v = 0.0f;
        ggml_vec_dot_f32(ew0, &v, (float*)params->wdata + i1 * ew0 * ne00 + (nh + k) * ew0,
                         (float*)params->wdata + ne02 * ew0 * ne00 + (i0 + nh + k) * ew0);

        dst_data[i0 / 2] += v;
      }
    }
  }
}

static void ggml_compute_forward_conv_1d_s2_ph(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_conv_1d_s2_ph_f16_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_conv_1d_s2_ph_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_conv_1d

static void ggml_compute_forward_conv_1d(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
  const int32_t p0 = ((const int32_t*)(dst->op_params))[1];
  const int32_t d0 = ((const int32_t*)(dst->op_params))[2];
  GGML_ASSERT(d0 == 1);                // dilation not supported
  GGML_ASSERT(p0 == src0->ne[0] / 2);  // only half padding supported
  if (s0 == 1) {
    ggml_compute_forward_conv_1d_s1_ph(params, src0, src1, dst);
  } else if (s0 == 2) {
    ggml_compute_forward_conv_1d_s2_ph(params, src0, src1, dst);
  } else {
    GGML_ASSERT(false);  // only stride 1 and 2 supported
  }
}

// ggml_compute_forward_conv_2d

static void ggml_compute_forward_conv_2d_f16_f32(const struct ggml_compute_params* params,
                                                 const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                 struct ggml_tensor* dst) {
  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk0 = ne00;
  const int nk1 = ne01;

  // size of the convolution row - the kernel size unrolled across all channels
  const int ew0 = nk0 * nk1 * ne02;

  const int32_t s0 = ((const int32_t*)(dst->op_params))[0];
  const int32_t s1 = ((const int32_t*)(dst->op_params))[1];
  const int32_t p0 = ((const int32_t*)(dst->op_params))[2];
  const int32_t p1 = ((const int32_t*)(dst->op_params))[3];
  const int32_t d0 = ((const int32_t*)(dst->op_params))[4];
  const int32_t d1 = ((const int32_t*)(dst->op_params))[5];

  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb10 == sizeof(float));

  if (params->type == GGML_TASK_INIT) {
    memset(params->wdata, 0, params->wsize);

    // prepare source data (src1)
    {
      ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + 0;

      for (int i12 = 0; i12 < ne12; i12++) {
        const float* const src = (float*)((char*)src1->data + i12 * nb12);
        ggml_fp16_t* dst_data = wdata;

        for (int i1 = 0; i1 < ne1; i1++) {
          for (int i0 = 0; i0 < ne0; i0++) {
            for (int ik1 = 0; ik1 < nk1; ik1++) {
              for (int ik0 = 0; ik0 < nk0; ik0++) {
                const int idx0 = i0 * s0 + ik0 * d0 - p0;
                const int idx1 = i1 * s1 + ik1 * d1 - p1;

                if (!(idx1 < 0 || idx1 >= ne11 || idx0 < 0 || idx0 >= ne10)) {
                  dst_data[(i1 * ne0 + i0) * ew0 + i12 * (nk0 * nk1) + ik1 * nk0 + ik0] =
                      GGML_FP32_TO_FP16(src[idx1 * ne10 + idx0]);
                }
              }
            }
          }
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // total patches in dst
  const int np = ne2;

  // patches per thread
  const int dp = (np + nth - 1) / nth;

  // patch range for this thread
  const int ip0 = dp * ith;
  const int ip1 = MIN(ip0 + dp, np);

  ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + 0;

  for (int i3 = 0; i3 < ne3; i3++) {
    for (int i2 = ip0; i2 < ip1; i2++) {
      float* dst_data = (float*)((char*)dst->data + i3 * nb3 + i2 * nb2);

      for (int i1 = 0; i1 < ne1; ++i1) {
        for (int i0 = 0; i0 < ne0; ++i0) {
          ggml_vec_dot_f16(ew0, dst_data + i1 * ne0 + i0, (ggml_fp16_t*)((char*)src0->data + i2 * nb03),
                           (ggml_fp16_t*)wdata + i3 * nb3 + (i1 * ne0 + i0) * ew0);
        }
      }
    }
  }
}

static void ggml_compute_forward_conv_2d(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         const struct ggml_tensor* src1, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_conv_2d_f16_f32(params, src0, src1, dst);
    } break;
    case GGML_TYPE_F32: {
      // ggml_compute_forward_conv_2d_f32(params, src0, src1, dst);
      GGML_ASSERT(false);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_conv_transpose_2d

static void ggml_compute_forward_conv_transpose_2d(const struct ggml_compute_params* params,
                                                   const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                   struct ggml_tensor* dst) {
  GGML_ASSERT(src0->type == GGML_TYPE_F16);
  GGML_ASSERT(src1->type == GGML_TYPE_F32);
  GGML_ASSERT(dst->type == GGML_TYPE_F32);

  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_BINARY_OP_LOCALS

  const int ith = params->ith;
  const int nth = params->nth;

  const int nk = ne00 * ne01 * ne02 * ne03;

  GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nb10 == sizeof(float));

  if (params->type == GGML_TASK_INIT) {
    memset(params->wdata, 0, params->wsize);

    // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
    {
      ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + 0;

      for (int64_t i03 = 0; i03 < ne03; i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
          const ggml_fp16_t* const src = (ggml_fp16_t*)((char*)src0->data + i03 * nb03 + i02 * nb02);
          ggml_fp16_t* dst_data = wdata + i02 * ne01 * ne00 * ne03;
          for (int64_t i01 = 0; i01 < ne01; i01++) {
            for (int64_t i00 = 0; i00 < ne00; i00++) {
              dst_data[i01 * ne00 * ne03 + i00 * ne03 + i03] = src[i01 * ne00 + i00];
            }
          }
        }
      }
    }

    // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
    {
      ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + nk;
      for (int i12 = 0; i12 < ne12; i12++) {
        for (int i11 = 0; i11 < ne11; i11++) {
          const float* const src = (float*)((char*)src1->data + i12 * nb12 + i11 * nb11);
          ggml_fp16_t* dst_data = wdata + i11 * ne10 * ne12;
          for (int i10 = 0; i10 < ne10; i10++) {
            dst_data[i10 * ne12 + i12] = GGML_FP32_TO_FP16(src[i10]);
          }
        }
      }
    }

    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int32_t stride = ggml_get_op_params_i32(dst, 0);

  // total patches in dst
  const int np = ne2;

  // patches per thread
  const int dp = (np + nth - 1) / nth;

  // patch range for this thread
  const int ip0 = dp * ith;
  const int ip1 = MIN(ip0 + dp, np);

  ggml_fp16_t* const wdata = (ggml_fp16_t*)params->wdata + 0;
  ggml_fp16_t* const wdata_src = wdata + nk;

  for (int i2 = ip0; i2 < ip1; i2++) {  // Cout
    float* dst_data = (float*)((char*)dst->data + i2 * nb2);
    ggml_fp16_t* wdata_kernel = wdata + i2 * ne01 * ne00 * ne03;
    for (int i11 = 0; i11 < ne11; i11++) {
      for (int i10 = 0; i10 < ne10; i10++) {
        const int i1n = i11 * ne10 * ne12 + i10 * ne12;
        for (int i01 = 0; i01 < ne01; i01++) {
          for (int i00 = 0; i00 < ne00; i00++) {
            float v = 0;
            ggml_vec_dot_f16(ne03, &v, wdata_src + i1n, wdata_kernel + i01 * ne00 * ne03 + i00 * ne03);
            dst_data[(i11 * stride + i01) * ne0 + i10 * stride + i00] += v;
          }
        }
      }
    }
  }
}

// ggml_compute_forward_pool_1d_sk_p0

static void ggml_compute_forward_pool_1d_sk_p0(const struct ggml_compute_params* params, const enum ggml_op_pool op,
                                               const struct ggml_tensor* src, const int k, struct ggml_tensor* dst) {
  assert(src->type == GGML_TYPE_F32);
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const char* cdata = (const char*)src->data;
  const char* const data_end = cdata + ggml_nbytes(src);
  float* drow = (float*)dst->data;

  const int64_t rs = dst->ne[0];

  while (cdata < data_end) {
    const float* const srow = (const float*)cdata;

    int j = 0;

    for (int64_t i = 0; i < rs; ++i) {
      switch (op) {
        case GGML_OP_POOL_AVG:
          drow[i] = 0;
          break;
        case GGML_OP_POOL_MAX:
          drow[i] = -FLT_MAX;
          break;
        case GGML_OP_POOL_COUNT:
          GGML_ASSERT(false);
          break;
      }
      for (int ki = 0; ki < k; ++ki) {
        switch (op) {
          case GGML_OP_POOL_AVG:
            drow[i] += srow[j];
            break;
          case GGML_OP_POOL_MAX:
            if (srow[j] > drow[i]) drow[i] = srow[j];
            break;
          case GGML_OP_POOL_COUNT:
            GGML_ASSERT(false);
            break;
        }
        ++j;
      }
      switch (op) {
        case GGML_OP_POOL_AVG:
          drow[i] /= k;
          break;
        case GGML_OP_POOL_MAX:
          break;
        case GGML_OP_POOL_COUNT:
          GGML_ASSERT(false);
          break;
      }
    }

    cdata += src->nb[1];
    drow += rs;
  }
}

// ggml_compute_forward_pool_1d

static void ggml_compute_forward_pool_1d(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  const int32_t* opts = (const int32_t*)dst->op_params;
  enum ggml_op_pool op = (ggml_op_pool)opts[0];
  const int k0 = opts[1];
  const int s0 = opts[2];
  const int p0 = opts[3];
  GGML_ASSERT(p0 == 0);   // padding not supported
  GGML_ASSERT(k0 == s0);  // only s = k supported

  ggml_compute_forward_pool_1d_sk_p0(params, op, src0, k0, dst);
}

// ggml_compute_forward_pool_2d_sk_p0

static void ggml_compute_forward_pool_2d_sk_p0(const struct ggml_compute_params* params, const enum ggml_op_pool op,
                                               const struct ggml_tensor* src, const int k0, const int k1,
                                               struct ggml_tensor* dst) {
  assert(src->type == GGML_TYPE_F32);
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const char* cdata = (const char*)src->data;
  const char* const data_end = cdata + ggml_nbytes(src);

  const int64_t px = dst->ne[0];
  const int64_t py = dst->ne[1];
  const int64_t pa = px * py;

  float* dplane = (float*)dst->data;

  const int ka = k0 * k1;

  while (cdata < data_end) {
    for (int oy = 0; oy < py; ++oy) {
      float* const drow = dplane + oy * px;
      for (int ox = 0; ox < px; ++ox) {
        float* const out = drow + ox;
        switch (op) {
          case GGML_OP_POOL_AVG:
            *out = 0;
            break;
          case GGML_OP_POOL_MAX:
            *out = -FLT_MAX;
            break;
          case GGML_OP_POOL_COUNT:
            GGML_ASSERT(false);
            break;
        }

        const int ix = ox * k0;
        const int iy = oy * k1;

        for (int ky = 0; ky < k1; ++ky) {
          const float* const srow = (const float*)(cdata + src->nb[1] * (iy + ky));
          for (int kx = 0; kx < k0; ++kx) {
            int j = ix + kx;
            switch (op) {
              case GGML_OP_POOL_AVG:
                *out += srow[j];
                break;
              case GGML_OP_POOL_MAX:
                if (srow[j] > *out) *out = srow[j];
                break;
              case GGML_OP_POOL_COUNT:
                GGML_ASSERT(false);
                break;
            }
          }
        }
        switch (op) {
          case GGML_OP_POOL_AVG:
            *out /= ka;
            break;
          case GGML_OP_POOL_MAX:
            break;
          case GGML_OP_POOL_COUNT:
            GGML_ASSERT(false);
            break;
        }
      }
    }

    cdata += src->nb[2];
    dplane += pa;
  }
}

// ggml_compute_forward_pool_2d

static void ggml_compute_forward_pool_2d(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  const int32_t* opts = (const int32_t*)dst->op_params;
  enum ggml_op_pool op = (ggml_op_pool)opts[0];
  const int k0 = opts[1];
  const int k1 = opts[2];
  const int s0 = opts[3];
  const int s1 = opts[4];
  const int p0 = opts[5];
  const int p1 = opts[6];
  GGML_ASSERT(p0 == 0);
  GGML_ASSERT(p1 == 0);  // padding not supported
  GGML_ASSERT(k0 == s0);
  GGML_ASSERT(k1 == s1);  // only s = k supported

  ggml_compute_forward_pool_2d_sk_p0(params, op, src0, k0, k1, dst);
}

// ggml_compute_forward_upscale

static void ggml_compute_forward_upscale_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                             struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_ASSERT(src0->nb[0] == sizeof(float));

  const int ith = params->ith;

  GGML_TENSOR_UNARY_OP_LOCALS

  const int scale_factor = dst->op_params[0];

  // TODO: optimize

  for (int i03 = 0; i03 < ne03; i03++) {
    for (int i02 = ith; i02 < ne02; i02++) {
      for (int m = 0; m < dst->ne[1]; m++) {
        int i01 = m / scale_factor;
        for (int n = 0; n < dst->ne[0]; n++) {
          int i00 = n / scale_factor;

          const float* x = (float*)((char*)src0->data + i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03);

          float* y = (float*)((char*)dst->data + n * dst->nb[0] + m * dst->nb[1] + i02 * dst->nb[2] + i03 * dst->nb[3]);

          *y = *x;
        }
      }
    }
  }
}

static void ggml_compute_forward_upscale(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                         struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_upscale_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_flash_attn

static void ggml_compute_forward_flash_attn_f32(const struct ggml_compute_params* params, const struct ggml_tensor* q,
                                                const struct ggml_tensor* k, const struct ggml_tensor* v,
                                                const bool masked, struct ggml_tensor* dst) {
  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = neq0;
  const int64_t N = neq1;
  const int64_t P = nek1 - N;
  const int64_t M = P + N;

  const int Mup = ggml_up(M, GGML_SOFT_MAX_UNROLL);

  GGML_ASSERT(ne0 == D);
  GGML_ASSERT(ne1 == N);
  GGML_ASSERT(P >= 0);

  GGML_ASSERT(nbq0 == sizeof(float));
  GGML_ASSERT(nbk0 == sizeof(float));
  GGML_ASSERT(nbv0 == sizeof(float));

  GGML_ASSERT(neq0 == D);
  GGML_ASSERT(nek0 == D);
  GGML_ASSERT(nev1 == D);

  GGML_ASSERT(neq1 == N);
  GGML_ASSERT(nek1 == N + P);
  GGML_ASSERT(nev1 == D);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  if (params->type == GGML_TASK_INIT) {
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by q rows using ggml_vec_dot_f32

  // total rows in q
  const int nr = neq1 * neq2 * neq3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const float scale = 1.0f / sqrtf(D);

  // printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int iq3 = ir / (neq2 * neq1);
    const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
    const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

    float* S = (float*)params->wdata + ith * (Mup + CACHE_LINE_SIZE_F32);

    for (int i = M; i < Mup; ++i) {
      S[i] = -INFINITY;
    }

    const int64_t masked_begin = masked ? (P + iq1 + 1) : M;
    for (int64_t ic = 0; ic < masked_begin; ++ic) {
      // k indices
      const int ik3 = iq3;
      const int ik2 = iq2 % nek2;
      const int ik1 = ic;

      // S indices
      const int i1 = ik1;

      ggml_vec_dot_f32(neq0, S + i1, (float*)((char*)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                       (float*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
    }

    // scale
    ggml_vec_scale_f32(masked_begin, S, scale);

    for (int64_t i = masked_begin; i < M; i++) {
      S[i] = -INFINITY;
    }

    // softmax
    // exclude known -INF S[..] values from max and loop
    // dont forget to set their SW values to zero
    {
      float max = -INFINITY;
      ggml_vec_max_f32(masked_begin, &max, S);

      ggml_float sum = 0.0;
      {
#ifdef GGML_SOFT_MAX_ACCELERATE
        max = -max;
        vDSP_vsadd(S, 1, &max, S, 1, Mup);
        vvexpf(S, S, &Mup);
        ggml_vec_sum_f32(Mup, &sum, S);
#else
        uint16_t scvt[GGML_SOFT_MAX_UNROLL];
        // UNUSED(scvt);
        ggml_float sump[GGML_SOFT_MAX_UNROLL] = {0.0};

        for (int i = 0; i < Mup; i += GGML_SOFT_MAX_UNROLL) {
          if (i >= masked_begin) {
            break;
          }
          float* SS = S + i;

          for (int j = 0; j < GGML_SOFT_MAX_UNROLL; ++j) {
            if (i + j >= masked_begin) {
              break;
            } else if (SS[j] == -INFINITY) {
              SS[j] = 0.0f;
            } else {
#ifndef GGML_FLASH_ATTN_EXP_FP16
              const float val = expf(SS[j] - max);
#else
              ggml_fp16_t s = GGML_FP32_TO_FP16(SS[j] - max);
              memcpy(&scvt[j], &s, sizeof(uint16_t));
              const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt[j]]);
#endif
              sump[j] += (ggml_float)val;
              SS[j] = val;
            }
          }
        }

        for (int i = 0; i < GGML_SOFT_MAX_UNROLL; i++) {
          sum += sump[i];
        }
#endif
      }

      assert(sum > 0.0);

      sum = 1.0 / sum;
      ggml_vec_scale_f32(masked_begin, S, sum);

#ifndef NDEBUG
      for (int i = 0; i < masked_begin; ++i) {
        assert(!isnan(S[i]));
        assert(!isinf(S[i]));
      }
#endif
    }

    for (int64_t ic = 0; ic < nev1; ++ic) {
      // dst indices
      const int i1 = iq1;
      const int i2 = iq2;
      const int i3 = iq3;

      // v indices
      const int iv2 = iq2 % nev2;
      const int iv3 = iq3;

      ggml_vec_dot_f32(masked_begin, (float*)((char*)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                       (float*)((char*)v->data + (ic * nbv1 + iv2 * nbv2 + iv3 * nbv3)), S);
    }
  }
}

static void ggml_compute_forward_flash_attn_f16(const struct ggml_compute_params* params, const struct ggml_tensor* q,
                                                const struct ggml_tensor* k, const struct ggml_tensor* v,
                                                const bool masked, struct ggml_tensor* dst) {
  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = neq0;
  const int64_t N = neq1;
  const int64_t P = nek1 - N;
  const int64_t M = P + N;

  const int Mup = ggml_up(M, GGML_SOFT_MAX_UNROLL);

  GGML_ASSERT(ne0 == D);
  GGML_ASSERT(ne1 == N);
  GGML_ASSERT(P >= 0);

  GGML_ASSERT(nbq0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nbk0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nbv0 == sizeof(ggml_fp16_t));

  GGML_ASSERT(neq0 == D);
  GGML_ASSERT(nek0 == D);
  GGML_ASSERT(nev1 == D);

  GGML_ASSERT(neq1 == N);
  GGML_ASSERT(nek1 == N + P);
  GGML_ASSERT(nev1 == D);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  if (params->type == GGML_TASK_INIT) {
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by q rows using ggml_vec_dot_f32

  // total rows in q
  const int nr = neq1 * neq2 * neq3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const float scale = 1.0f / sqrtf(D);

  // printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int iq3 = ir / (neq2 * neq1);
    const int iq2 = (ir - iq3 * neq2 * neq1) / neq1;
    const int iq1 = (ir - iq3 * neq2 * neq1 - iq2 * neq1);

    float* S = (float*)params->wdata + ith * (2 * Mup + CACHE_LINE_SIZE_F32);

    for (int i = M; i < Mup; ++i) {
      S[i] = -INFINITY;
    }

    if (GGML_VEC_DOT_UNROLL > 2 || nek1 % GGML_VEC_DOT_UNROLL != 0) {
      for (int64_t ic = 0; ic < nek1; ++ic) {
        // k indices
        const int ik3 = iq3;
        const int ik2 = iq2 % nek2;
        const int ik1 = ic;

        // S indices
        const int i1 = ik1;

        ggml_vec_dot_f16(neq0, S + i1, (ggml_fp16_t*)((char*)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                         (ggml_fp16_t*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
      }
    } else {
      for (int64_t ic = 0; ic < nek1; ic += GGML_VEC_DOT_UNROLL) {
        // k indices
        const int ik3 = iq3;
        const int ik2 = iq2 % nek2;
        const int ik1 = ic;

        // S indices
        const int i1 = ik1;

        ggml_vec_dot_f16_unroll(neq0, nbk1, S + i1, ((char*)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                                (ggml_fp16_t*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
      }
    }

    // scale
    ggml_vec_scale_f32(nek1, S, scale);

    if (masked) {
      for (int64_t i = P; i < M; i++) {
        if (i > P + iq1) {
          S[i] = -INFINITY;
        }
      }
    }

    // softmax
    // todo: exclude known -INF S[..] values from max and loop, assuming their results to be zero.
    // dont forget to set their S values to zero
    {
      float max = -INFINITY;
      ggml_vec_max_f32(M, &max, S);

      ggml_float sum = 0.0;
      {
#ifdef GGML_SOFT_MAX_ACCELERATE
        max = -max;
        vDSP_vsadd(S, 1, &max, S, 1, Mup);
        vvexpf(S, S, &Mup);
        ggml_vec_sum_f32(Mup, &sum, S);
#else
        uint16_t scvt[GGML_SOFT_MAX_UNROLL];
        ggml_float sump[GGML_SOFT_MAX_UNROLL] = {0.0};

        for (int i = 0; i < Mup; i += GGML_SOFT_MAX_UNROLL) {
          float* SS = S + i;

          for (int j = 0; j < GGML_SOFT_MAX_UNROLL; ++j) {
            if (SS[j] == -INFINITY) {
              SS[j] = 0.0f;
            } else {
              ggml_fp16_t s = GGML_FP32_TO_FP16(SS[j] - max);
              memcpy(&scvt[j], &s, sizeof(uint16_t));
              const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt[j]]);
              sump[j] += (ggml_float)val;
              SS[j] = val;
            }
          }
        }

        for (int i = 0; i < GGML_SOFT_MAX_UNROLL; i++) {
          sum += sump[i];
        }
#endif
      }

      assert(sum > 0.0);

      sum = 1.0 / sum;
      ggml_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
      for (int i = 0; i < M; ++i) {
        assert(!isnan(S[i]));
        assert(!isinf(S[i]));
      }
#endif
    }

    ggml_fp16_t* S16 = (ggml_fp16_t*)((float*)params->wdata + ith * (2 * Mup + CACHE_LINE_SIZE_F32) + Mup);

    for (int64_t i = 0; i < M; i++) {
      S16[i] = GGML_FP32_TO_FP16(S[i]);
    }

    // todo: exclude known zero S[..] values from dot (reducing nev0 and increasing begin of v and S16).
    if (GGML_VEC_DOT_UNROLL == 1 || (nev1 % GGML_VEC_DOT_UNROLL != 0)) {
      for (int64_t ic = 0; ic < nev1; ++ic) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // v indices
        const int iv2 = iq2 % nev2;
        const int iv3 = iq3;

        ggml_vec_dot_f16(nev0, (float*)((char*)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                         (ggml_fp16_t*)((char*)v->data + (ic * nbv1 + iv2 * nbv2 + iv3 * nbv3)), S16);
      }
    } else {
      for (int64_t ic = 0; ic < nev1; ic += GGML_VEC_DOT_UNROLL) {
        // dst indices
        const int i1 = iq1;
        const int i2 = iq2;
        const int i3 = iq3;

        // v indices
        const int iv2 = iq2 % nev2;
        const int iv3 = iq3;

        ggml_vec_dot_f16_unroll(nev0, nbv1, (float*)((char*)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                                ((char*)v->data + (ic * nbv1 + iv2 * nbv2 + iv3 * nbv3)), S16);
      }
    }
  }
}

static void ggml_compute_forward_flash_attn(const struct ggml_compute_params* params, const struct ggml_tensor* q,
                                            const struct ggml_tensor* k, const struct ggml_tensor* v, const bool masked,
                                            struct ggml_tensor* dst) {
  switch (q->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_flash_attn_f16(params, q, k, v, masked, dst);
    } break;
    case GGML_TYPE_F32: {
      ggml_compute_forward_flash_attn_f32(params, q, k, v, masked, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_flash_ff

static void ggml_compute_forward_flash_ff_f16(const struct ggml_compute_params* params,
                                              const struct ggml_tensor* a,   // F16
                                              const struct ggml_tensor* b0,  // F16 fc_w
                                              const struct ggml_tensor* b1,  // F32 fc_b
                                              const struct ggml_tensor* c0,  // F16 proj_w
                                              const struct ggml_tensor* c1,  // F32 proj_b
                                              struct ggml_tensor* dst) {
  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_LOCALS(int64_t, nea, a, ne)
  GGML_TENSOR_LOCALS(size_t, nba, a, nb)
  GGML_TENSOR_LOCALS(int64_t, neb0, b0, ne)
  GGML_TENSOR_LOCALS(size_t, nbb0, b0, nb)
  GGML_TENSOR_LOCALS(int64_t, neb1, b1, ne)
  GGML_TENSOR_LOCALS(size_t, nbb1, b1, nb)
  GGML_TENSOR_LOCALS(int64_t, nec0, c0, ne)
  GGML_TENSOR_LOCALS(size_t, nbc0, c0, nb)
  GGML_TENSOR_LOCALS(int64_t, nec1, c1, ne)
  GGML_TENSOR_LOCALS(size_t, nbc1, c1, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = nea0;
  // const int64_t N = nea1;
  const int64_t M = neb01;

  GGML_ASSERT(ne0 == nea0);
  GGML_ASSERT(ne1 == nea1);
  GGML_ASSERT(ne2 == nea2);

  GGML_ASSERT(nba0 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nbb00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nbb10 == sizeof(float));
  GGML_ASSERT(nbc00 == sizeof(ggml_fp16_t));
  GGML_ASSERT(nbc10 == sizeof(float));

  GGML_ASSERT(neb00 == D);
  GGML_ASSERT(neb01 == M);
  GGML_ASSERT(neb10 == M);
  GGML_ASSERT(neb11 == 1);

  GGML_ASSERT(nec00 == M);
  GGML_ASSERT(nec01 == D);
  GGML_ASSERT(nec10 == D);
  GGML_ASSERT(nec11 == 1);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  if (params->type == GGML_TASK_INIT) {
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // parallelize by a rows using ggml_vec_dot_f32

  // total rows in a
  const int nr = nea1 * nea2 * nea3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int ir = ir0; ir < ir1; ++ir) {
    // a indices
    const int ia3 = ir / (nea2 * nea1);
    const int ia2 = (ir - ia3 * nea2 * nea1) / nea1;
    const int ia1 = (ir - ia3 * nea2 * nea1 - ia2 * nea1);

    float* S = (float*)params->wdata + ith * (2 * M + CACHE_LINE_SIZE_F32);

    for (int64_t ic = 0; ic < neb01; ++ic) {
      // b0 indices
      const int ib03 = ia3;
      const int ib02 = ia2;
      const int ib01 = ic;

      // S indices
      const int i1 = ib01;

      ggml_vec_dot_f16(nea0, S + i1, (ggml_fp16_t*)((char*)b0->data + (ib01 * nbb01 + ib02 * nbb02 + ib03 * nbb03)),
                       (ggml_fp16_t*)((char*)a->data + (ia1 * nba1 + ia2 * nba2 + ia3 * nba3)));
    }

    ggml_vec_add_f32(neb01, S, S, (float*)b1->data);
    // ggml_vec_gelu_f32(neb01, S, S);

    ggml_fp16_t* S16 = (ggml_fp16_t*)((float*)params->wdata + ith * (2 * M + CACHE_LINE_SIZE_F32) + M);

    for (int64_t i = 0; i < M; i++) {
      S16[i] = GGML_FP32_TO_FP16(S[i]);
    }

    ggml_vec_gelu_f16(neb01, S16, S16);

    {
      // dst indices
      const int i1 = ia1;
      const int i2 = ia2;
      const int i3 = ia3;

      for (int64_t ic = 0; ic < nec01; ++ic) {
        ggml_vec_dot_f16(neb01, (float*)((char*)dst->data + (ic * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3)),
                         (ggml_fp16_t*)((char*)c0->data + (ic * nbc01 + i2 * nbc02 + i3 * nbc03)), S16);
      }

      ggml_vec_add_f32(nec01, (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3)),
                       (float*)((char*)dst->data + (i1 * nb1 + i2 * nb2 + i3 * nb3)), (float*)c1->data);
    }
  }
}

static void ggml_compute_forward_flash_ff(const struct ggml_compute_params* params, const struct ggml_tensor* a,
                                          const struct ggml_tensor* b0, const struct ggml_tensor* b1,
                                          const struct ggml_tensor* c0, const struct ggml_tensor* c1,
                                          struct ggml_tensor* dst) {
  switch (b0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_flash_ff_f16(params, a, b0, b1, c0, c1, dst);
    } break;
    case GGML_TYPE_F32: {
      GGML_ASSERT(false);  // TODO
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_flash_attn_back

static void ggml_compute_forward_flash_attn_back_f32(const struct ggml_compute_params* params,
                                                     const struct ggml_tensor* q, const struct ggml_tensor* k,
                                                     const struct ggml_tensor* v, const struct ggml_tensor* d,
                                                     const bool masked, struct ggml_tensor* dst) {
  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  GGML_TENSOR_LOCALS(int64_t, neq, q, ne)
  GGML_TENSOR_LOCALS(size_t, nbq, q, nb)
  GGML_TENSOR_LOCALS(int64_t, nek, k, ne)
  GGML_TENSOR_LOCALS(size_t, nbk, k, nb)
  GGML_TENSOR_LOCALS(int64_t, nev, v, ne)
  GGML_TENSOR_LOCALS(size_t, nbv, v, nb)
  GGML_TENSOR_LOCALS(int64_t, ned, d, ne)
  GGML_TENSOR_LOCALS(size_t, nbd, d, nb)
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
  GGML_TENSOR_LOCALS(size_t, nb, dst, nb)

  const int ith = params->ith;
  const int nth = params->nth;

  const int64_t D = neq0;
  const int64_t N = neq1;
  const int64_t P = nek1 - N;
  const int64_t M = P + N;

  const int Mup = ggml_up(M, GGML_SOFT_MAX_UNROLL);
  const int mxDM = MAX(D, Mup);

  // GGML_ASSERT(ne0 == D);
  // GGML_ASSERT(ne1 == N);
  GGML_ASSERT(P >= 0);

  GGML_ASSERT(nbq0 == sizeof(float));
  GGML_ASSERT(nbk0 == sizeof(float));
  GGML_ASSERT(nbv0 == sizeof(float));

  GGML_ASSERT(neq0 == D);
  GGML_ASSERT(nek0 == D);
  GGML_ASSERT(nev1 == D);
  GGML_ASSERT(ned0 == D);

  GGML_ASSERT(neq1 == N);
  GGML_ASSERT(nek1 == N + P);
  GGML_ASSERT(nev1 == D);
  GGML_ASSERT(ned1 == N);

  // dst cannot be transposed or permuted
  GGML_ASSERT(nb0 == sizeof(float));
  GGML_ASSERT(nb0 <= nb1);
  GGML_ASSERT(nb1 <= nb2);
  GGML_ASSERT(nb2 <= nb3);

  if (params->type == GGML_TASK_INIT) {
    if (ith == 0) {
      memset(dst->data, 0, nb0 * ne0 * ne1 * ne2 * ne3);
    }
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int64_t elem_q = ggml_nelements(q);
  const int64_t elem_k = ggml_nelements(k);

  enum ggml_type result_type = dst->type;
  GGML_ASSERT(type_traits[result_type].blck_size == 1);
  const size_t tsize = type_traits[result_type].type_size;

  const size_t offs_q = 0;
  const size_t offs_k = offs_q + GGML_PAD(elem_q * tsize, GGML_MEM_ALIGN);
  const size_t offs_v = offs_k + GGML_PAD(elem_k * tsize, GGML_MEM_ALIGN);

  void* grad_q = (char*)dst->data;
  void* grad_k = (char*)dst->data + offs_k;
  void* grad_v = (char*)dst->data + offs_v;

  const size_t nbgq1 = nb0 * neq0;
  const size_t nbgq2 = nb0 * neq0 * neq1;
  const size_t nbgq3 = nb0 * neq0 * neq1 * neq2;

  const size_t nbgk1 = nb0 * nek0;
  const size_t nbgk2 = nb0 * nek0 * nek1;
  const size_t nbgk3 = nb0 * nek0 * nek1 * neq2;

  const size_t nbgv1 = nb0 * nev0;
  const size_t nbgv2 = nb0 * nev0 * nev1;
  const size_t nbgv3 = nb0 * nev0 * nev1 * neq2;

  // parallelize by k rows using ggml_vec_dot_f32

  // total rows in k
  const int nr = nek2 * nek3;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  const float scale = 1.0f / sqrtf(D);

  // printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

  // how often k2 (and v2) is repeated in q2
  int nrep = neq2 / nek2;

  for (int ir = ir0; ir < ir1; ++ir) {
    // q indices
    const int ik3 = ir / (nek2);
    const int ik2 = ir - ik3 * nek2;

    const int iq3 = ik3;
    const int id3 = ik3;
    const int iv3 = ik3;
    const int iv2 = ik2;

    for (int irep = 0; irep < nrep; ++irep) {
      const int iq2 = ik2 + irep * nek2;
      const int id2 = iq2;

      // (ik2 + irep*nek2) % nek2 == ik2
      for (int iq1 = 0; iq1 < neq1; ++iq1) {
        const int id1 = iq1;

        // not sure about CACHE_LINE_SIZE_F32..
        // - maybe it must not be multiplied by 2 and excluded from .. in SM 1*(..) offset?
        float* S = (float*)params->wdata + ith * 2 * (mxDM + CACHE_LINE_SIZE_F32) + 0 * (mxDM + CACHE_LINE_SIZE_F32);
        float* SM = (float*)params->wdata + ith * 2 * (mxDM + CACHE_LINE_SIZE_F32) + 1 * (mxDM + CACHE_LINE_SIZE_F32);

        for (int i = M; i < Mup; ++i) {
          S[i] = -INFINITY;
        }

        const int64_t masked_begin = masked ? (P + iq1 + 1) : M;
        for (int64_t ic = 0; ic < masked_begin; ++ic) {
          // k indices
          const int ik1 = ic;

          // S indices
          const int i1 = ik1;

          ggml_vec_dot_f32(neq0, S + i1, (float*)((char*)k->data + (ik1 * nbk1 + ik2 * nbk2 + ik3 * nbk3)),
                           (float*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)));
        }

        // scale
        ggml_vec_scale_f32(masked_begin, S, scale);

        for (int64_t i = masked_begin; i < M; i++) {
          S[i] = -INFINITY;
        }

        // softmax
        // exclude known -INF S[..] values from max and loop
        // dont forget to set their SM values to zero
        {
          float max = -INFINITY;
          ggml_vec_max_f32(masked_begin, &max, S);

          ggml_float sum = 0.0;
          {
#ifdef GGML_SOFT_MAX_ACCELERATE
            max = -max;
            vDSP_vsadd(SM, 1, &max, SM, 1, Mup);
            vvexpf(SM, SM, &Mup);
            ggml_vec_sum_f32(Mup, &sum, SM);
#else
            uint16_t scvt[GGML_SOFT_MAX_UNROLL];
            // UNUSED(scvt);
            ggml_float sump[GGML_SOFT_MAX_UNROLL] = {0.0};

            for (int i = 0; i < Mup; i += GGML_SOFT_MAX_UNROLL) {
              if (i >= masked_begin) {
                break;
              }
              float* SR = S + i;
              float* SW = SM + i;

              for (int j = 0; j < GGML_SOFT_MAX_UNROLL; ++j) {
                if (i + j >= masked_begin) {
                  break;
                } else if (SR[j] == -INFINITY) {
                  SW[j] = 0.0f;
                } else {
#ifndef GGML_FLASH_ATTN_EXP_FP16
                  const float val = expf(SR[j] - max);
#else
                  ggml_fp16_t s = GGML_FP32_TO_FP16(SR[j] - max);
                  memcpy(&scvt[j], &s, sizeof(uint16_t));
                  const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt[j]]);
#endif
                  sump[j] += (ggml_float)val;
                  SW[j] = val;
                }
              }
            }

            for (int i = 0; i < GGML_SOFT_MAX_UNROLL; i++) {
              sum += sump[i];
            }
#endif
          }

          assert(sum > 0.0);

          sum = 1.0 / sum;
          ggml_vec_scale_f32(masked_begin, SM, sum);
        }

        // step-by-step explanation
        {
          // forward-process                    shape      grads from backward process
          // parallel_for ik2,ik3:
          //  for irep:
          //   iq2 = ik2 + irep*nek2
          //   k[:D,:M,:,:]                     [D,M,:,:]  grad[k][:D,:M,ik2,ik3]  += grad[kcur]
          //   q[:D,:N,:,:]                     [D,N,:,:]  grad[q][:D,iq1,iq2,iq3] += grad[qcur]
          //   v[:M,:D,:,:]                     [M,D,:,:]  grad[v][:M,:D,iv2,iv3]  += grad[vcur]
          //   for iq1:
          //    kcur   = k[:D,:M,ik2,ik3]       [D,M,1,1]  grad[kcur] = grad[S1].T @ qcur
          //    qcur   = q[:D,iq1,iq2,iq3]      [D,1,1,1]  grad[qcur] = grad[S1]   @ kcur
          //    vcur   = v[:M,:D,iv2,iv3]       [M,D,1,1]  grad[vcur] = grad[S5].T @ S4
          //    S0     = -Inf                   [D,1,1,1]
          //   ~S1[i]  = dot(kcur[:D,i], qcur)
          //    S1     = qcur @ kcur.T          [M,1,1,1]  grad[S1]   = grad[S2] * scale
          //    S2     = S1 * scale             [M,1,1,1]  grad[S2]   = diag_mask_zero(grad[S3], P)
          //    S3     = diag_mask_inf(S2, P)   [M,1,1,1]  grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
          //    S4     = softmax(S3)            [M,1,1,1]  grad[S4]   = grad[S5] @ vcur
          //   ~S5[i]  = dot(vcur[:,i], S4)
          //    S5     = S4 @ vcur.T            [D,1,1,1]  grad[S5]   = d[:D,id1,id2,id3]
          //   ~dst[i,iq1,iq2,iq3]  = S5[i]              ^
          //    dst[:D,iq1,iq2,iq3] = S5                 | grad[dst[:D,iq1,iq2,iq3]] = d[:D,id1,id2,id3]
          // dst                               backward-/ grad[dst]                 = d
          //
          // output gradients with their dependencies:
          //
          // grad[kcur] = grad[S1].T @ qcur
          // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
          // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
          // grad[S4]   = grad[S5] @ vcur
          // grad[S4]   = d[:D,id1,id2,id3] @ vcur
          // grad[qcur] = grad[S1]   @ kcur
          // grad[vcur] = grad[S5].T @ S4
          // grad[vcur] = d[:D,id1,id2,id3].T @ S4
          //
          // in post-order:
          //
          // S1         = qcur @ kcur.T
          // S2         = S1 * scale
          // S3         = diag_mask_inf(S2, P)
          // S4         = softmax(S3)
          // grad[S4]   = d[:D,id1,id2,id3] @ vcur
          // grad[S3]   = S4 * (grad[S4] - dot(S4, grad[S4]))
          // grad[S1]   = diag_mask_zero(grad[S3], P) * scale
          // grad[qcur] = grad[S1]   @ kcur
          // grad[kcur] = grad[S1].T @ qcur
          // grad[vcur] = d[:D,id1,id2,id3].T @ S4
          //
          // using less variables (SM=S4):
          //
          // S             = diag_mask_inf(qcur @ kcur.T * scale, P)
          // SM            = softmax(S)
          // S             = d[:D,iq1,iq2,iq3] @ vcur
          // dot_SM_gradSM = dot(SM, S)
          // S             = SM * (S - dot(SM, S))
          // S             = diag_mask_zero(S, P) * scale
          //
          // grad[q][:D,iq1,iq2,iq3] += S   @ kcur
          // grad[k][:D,:M,ik2,ik3]  += S.T @ qcur
          // grad[v][:M,:D,iv2,iv3]  += d[:D,id1,id2,id3].T @ SM
        }

        // S = gradSM = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
        // S = d[:D,id1,id2,id3] @ vcur[:,:,iv2,iv3]
        // for ic:
        //   S[:M] += vcur[:M,ic,iv2,iv3] * d[ic,id1,id2,id3]
        // exclude known future zero S[..] values from operation
        ggml_vec_set_f32(masked_begin, S, 0);
        for (int64_t ic = 0; ic < D; ++ic) {
          ggml_vec_mad_f32(masked_begin, S, (float*)((char*)v->data + (ic * nbv1 + iv2 * nbv2 + iv3 * nbv3)),
                           *(float*)((char*)d->data + (ic * nbd0 + id1 * nbd1 + id2 * nbd2 + id3 * nbd3)));
        }

        // S = SM * (S - dot(SM, S))
        float dot_SM_gradSM = 0;
        ggml_vec_dot_f32(masked_begin, &dot_SM_gradSM, SM, S);
        ggml_vec_acc1_f32(M, S, -dot_SM_gradSM);
        ggml_vec_mul_f32(masked_begin, S, S, SM);

        // S = diag_mask_zero(S, P) * scale
        // already done by above ggml_vec_set_f32

        // exclude known zero S[..] values from operation
        ggml_vec_scale_f32(masked_begin, S, scale);

        // S    shape [M,1]
        // SM   shape [M,1]
        // kcur shape [D,M]
        // qcur shape [D,1]
        // vcur shape [M,D]

        // grad[q][:D,iq1,iq2,iq3] += S @ kcur
        // grad[q][:D,iq1,iq2,iq3] += shape[M,1] @ shape[D,M]
        // for ic:
        //  grad[q][:D,iq1,iq2,iq3] += S[ic] * kcur[:D,ic,ik2,ik3]
        // exclude known zero S[..] values from loop
        for (int64_t ic = 0; ic < masked_begin; ++ic) {
          ggml_vec_mad_f32(D, (float*)((char*)grad_q + (iq1 * nbgq1 + iq2 * nbgq2 + iq3 * nbgq3)),
                           (float*)((char*)k->data + (ic * nbk1 + ik2 * nbk2 + ik3 * nbk3)), S[ic]);
        }

        // grad[k][:D,:M,iq2,iq3] += S.T @ qcur
        // for ic:
        //  grad[k][:D,ic,iq2,iq3] += S.T[0,ic] * qcur[:D,0]
        //  grad[k][:D,ic,iq2,iq3] += S[ic]     * qcur[:D,0]
        // exclude known zero S[..] values from loop
        for (int64_t ic = 0; ic < masked_begin; ++ic) {
          ggml_vec_mad_f32(D, (float*)((char*)grad_k + (ic * nbgk1 + ik2 * nbgk2 + ik3 * nbgk3)),
                           (float*)((char*)q->data + (iq1 * nbq1 + iq2 * nbq2 + iq3 * nbq3)), S[ic]);
        }

        // grad[v][:M,:D,iv2,iv3] += d[:D,id1,id2,id3].T       @ SM
        // for ic:
        //  grad[v][:M,ic,iv2,iv3] += d[:D,id1,id2,id3].T[0,ic] * SM[:M]
        //  grad[v][:M,ic,iv2,iv3] += d[ic,id1,id2,id3]         * SM[:M]
        // exclude known zero SM[..] values from mad
        for (int64_t ic = 0; ic < D; ++ic) {
          ggml_vec_mad_f32(masked_begin, (float*)((char*)grad_v + (ic * nbgv1 + iv2 * nbgv2 + iv3 * nbgv3)), SM,
                           *(float*)((char*)d->data + (ic * nbd0 + id1 * nbd1 + id2 * nbd2 + id3 * nbd3)));
        }
      }
    }
  }
}

static void ggml_compute_forward_flash_attn_back(const struct ggml_compute_params* params, const struct ggml_tensor* q,
                                                 const struct ggml_tensor* k, const struct ggml_tensor* v,
                                                 const struct ggml_tensor* d, const bool masked,
                                                 struct ggml_tensor* dst) {
  switch (q->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_flash_attn_back_f32(params, q, k, v, d, masked, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_win_part

static void ggml_compute_forward_win_part_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                              struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)

  const int32_t nep0 = ((const int32_t*)(dst->op_params))[0];
  const int32_t nep1 = ((const int32_t*)(dst->op_params))[1];
  const int32_t w = ((const int32_t*)(dst->op_params))[2];

  assert(ne00 == ne0);
  assert(ne3 == nep0 * nep1);

  // TODO: optimize / multi-thread
  for (int py = 0; py < nep1; ++py) {
    for (int px = 0; px < nep0; ++px) {
      const int64_t i3 = py * nep0 + px;
      for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
          for (int64_t i0 = 0; i0 < ne0; ++i0) {
            const int64_t i02 = py * w + i2;
            const int64_t i01 = px * w + i1;
            const int64_t i00 = i0;

            const int64_t i = i3 * ne2 * ne1 * ne0 + i2 * ne1 * ne0 + i1 * ne0 + i0;
            const int64_t j = i02 * ne01 * ne00 + i01 * ne00 + i00;

            if (py * w + i2 >= ne02 || px * w + i1 >= ne01) {
              ((float*)dst->data)[i] = 0.0f;
            } else {
              ((float*)dst->data)[i] = ((float*)src0->data)[j];
            }
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_win_part(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                          struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_win_part_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_win_unpart

static void ggml_compute_forward_win_unpart_f32(const struct ggml_compute_params* params,
                                                const struct ggml_tensor* src0, struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne)
  GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)

  const int32_t w = ((const int32_t*)(dst->op_params))[0];

  // padding
  const int px = (w - ne1 % w) % w;
  // const int py = (w - ne2%w)%w;

  const int npx = (px + ne1) / w;
  // const int npy = (py + ne2)/w;

  assert(ne0 == ne00);

  // TODO: optimize / multi-thread
  for (int64_t i2 = 0; i2 < ne2; ++i2) {
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
      for (int64_t i0 = 0; i0 < ne0; ++i0) {
        const int ip2 = i2 / w;
        const int ip1 = i1 / w;

        const int64_t i02 = i2 % w;
        const int64_t i01 = i1 % w;
        const int64_t i00 = i0;

        const int64_t i = (ip2 * npx + ip1) * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00 + i00;
        const int64_t j = i2 * ne1 * ne0 + i1 * ne0 + i0;

        ((float*)dst->data)[j] = ((float*)src0->data)[i];
      }
    }
  }
}

static void ggml_compute_forward_win_unpart(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_win_unpart_f32(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// gmml_compute_forward_unary

static void ggml_compute_forward_unary(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                       struct ggml_tensor* dst) {
  const enum ggml_unary_op op = ggml_get_unary_op(dst);

  switch (op) {
    case GGML_UNARY_OP_ABS: {
      ggml_compute_forward_abs(params, src0, dst);
    } break;
    case GGML_UNARY_OP_SGN: {
      ggml_compute_forward_sgn(params, src0, dst);
    } break;
    case GGML_UNARY_OP_NEG: {
      ggml_compute_forward_neg(params, src0, dst);
    } break;
    case GGML_UNARY_OP_STEP: {
      ggml_compute_forward_step(params, src0, dst);
    } break;
    case GGML_UNARY_OP_TANH: {
      ggml_compute_forward_tanh(params, src0, dst);
    } break;
    case GGML_UNARY_OP_ELU: {
      ggml_compute_forward_elu(params, src0, dst);
    } break;
    case GGML_UNARY_OP_RELU: {
      ggml_compute_forward_relu(params, src0, dst);
    } break;
    case GGML_UNARY_OP_GELU: {
      ggml_compute_forward_gelu(params, src0, dst);
    } break;
    case GGML_UNARY_OP_GELU_QUICK: {
      ggml_compute_forward_gelu_quick(params, src0, dst);
    } break;
    case GGML_UNARY_OP_SILU: {
      ggml_compute_forward_silu(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_get_rel_pos

static void ggml_compute_forward_get_rel_pos_f16(const struct ggml_compute_params* params,
                                                 const struct ggml_tensor* src0, struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  // ref:
  // https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L292-L322

  GGML_TENSOR_UNARY_OP_LOCALS

  const int64_t w = ne1;

  ggml_fp16_t* src0_data = (ggml_fp16_t*)src0->data;
  ggml_fp16_t* dst_data = (ggml_fp16_t*)dst->data;

  for (int64_t i2 = 0; i2 < ne2; ++i2) {
    for (int64_t i1 = 0; i1 < ne1; ++i1) {
      const int64_t pos = (w - i1 - 1) + i2;
      for (int64_t i0 = 0; i0 < ne0; ++i0) {
        dst_data[i2 * ne1 * ne0 + i1 * ne0 + i0] = src0_data[pos * ne00 + i0];
      }
    }
  }
}

static void ggml_compute_forward_get_rel_pos(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                             struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F16: {
      ggml_compute_forward_get_rel_pos_f16(params, src0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_add_rel_pos

static void ggml_compute_forward_add_rel_pos_f32(const struct ggml_compute_params* params,
                                                 const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                 const struct ggml_tensor* src2, struct ggml_tensor* dst) {
  const bool inplace = (bool)((int32_t*)dst->op_params)[0];
  if (!inplace && params->type == GGML_TASK_INIT) {
    memcpy((char*)dst->data, (char*)src0->data, ggml_nbytes(dst));
    return;
  }
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  int64_t t0 = ggml_perf_time_us();
  // UNUSED(t0);

  // ref:
  // https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359

  float* src1_data = (float*)src1->data;
  float* src2_data = (float*)src2->data;
  float* dst_data = (float*)dst->data;

  const int64_t ne10 = src1->ne[0];
  const int64_t ne11 = src1->ne[1];
  const int64_t ne12 = src1->ne[2];
  const int64_t ne13 = src1->ne[3];

  const int ith = params->ith;
  const int nth = params->nth;

  // total patches in dst
  const int np = ne13;

  // patches per thread
  const int dp = (np + nth - 1) / nth;

  // patch range for this thread
  const int ip0 = dp * ith;
  const int ip1 = MIN(ip0 + dp, np);

  for (int64_t i13 = ip0; i13 < ip1; ++i13) {
    for (int64_t i12 = 0; i12 < ne12; ++i12) {
      for (int64_t i11 = 0; i11 < ne11; ++i11) {
        const int64_t jp1 = i13 * ne12 * ne11 * ne10 + i12 * ne11 * ne10 + i11 * ne10;
        for (int64_t i10 = 0; i10 < ne10; ++i10) {
          const int64_t jp0 = jp1 + i10;
          const float src1_e = src1_data[jp0];
          const float src2_e = src2_data[jp0];

          const int64_t jdh = jp0 * ne10;
          const int64_t jdw = jdh - (ne10 - 1) * i10;

          for (int64_t j = 0; j < ne10; ++j) {
            dst_data[jdh + j] += src2_e;
            dst_data[jdw + j * ne10] += src1_e;
          }
        }
      }
    }
  }
}

static void ggml_compute_forward_add_rel_pos(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                             const struct ggml_tensor* src1, const struct ggml_tensor* src2,
                                             struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_add_rel_pos_f32(params, src0, src1, src2, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_map_unary

static void ggml_compute_forward_map_unary_f32(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                               struct ggml_tensor* dst, const ggml_unary_op_f32_t fun) {
  GGML_ASSERT(ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    fun(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])));
  }
}

static void ggml_compute_forward_map_unary(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                           struct ggml_tensor* dst, const ggml_unary_op_f32_t fun) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_map_unary_f32(params, src0, dst, fun);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_map_binary

static void ggml_compute_forward_map_binary_f32(const struct ggml_compute_params* params,
                                                const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                struct ggml_tensor* dst, const ggml_binary_op_f32_t fun) {
  assert(params->ith == 0);
  assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const int n = ggml_nrows(src0);
  const int nc = src0->ne[0];

  assert(dst->nb[0] == sizeof(float));
  assert(src0->nb[0] == sizeof(float));
  assert(src1->nb[0] == sizeof(float));

  for (int i = 0; i < n; i++) {
    fun(nc, (float*)((char*)dst->data + i * (dst->nb[1])), (float*)((char*)src0->data + i * (src0->nb[1])),
        (float*)((char*)src1->data + i * (src1->nb[1])));
  }
}

static void ggml_compute_forward_map_binary(const struct ggml_compute_params* params, const struct ggml_tensor* src0,
                                            const struct ggml_tensor* src1, struct ggml_tensor* dst,
                                            const ggml_binary_op_f32_t fun) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_map_binary_f32(params, src0, src1, dst, fun);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_map_custom1

static void ggml_compute_forward_map_custom1_f32(const struct ggml_compute_params* params, const struct ggml_tensor* a,
                                                 struct ggml_tensor* dst, const ggml_custom1_op_f32_t fun) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  fun(dst, a);
}

// ggml_compute_forward_map_custom2

static void ggml_compute_forward_map_custom2_f32(const struct ggml_compute_params* params, const struct ggml_tensor* a,
                                                 const struct ggml_tensor* b, struct ggml_tensor* dst,
                                                 const ggml_custom2_op_f32_t fun) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  fun(dst, a, b);
}

// ggml_compute_forward_map_custom3

static void ggml_compute_forward_map_custom3_f32(const struct ggml_compute_params* params, const struct ggml_tensor* a,
                                                 const struct ggml_tensor* b, const struct ggml_tensor* c,
                                                 struct ggml_tensor* dst, const ggml_custom3_op_f32_t fun) {
  assert(params->ith == 0);

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  fun(dst, a, b, c);
}

// ggml_compute_forward_map_custom1

static void ggml_compute_forward_map_custom1(const struct ggml_compute_params* params, const struct ggml_tensor* a,
                                             struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  struct ggml_map_custom1_op_params* p = (struct ggml_map_custom1_op_params*)dst->op_params;

  p->fun(dst, a, params->ith, params->nth, p->userdata);
}

// ggml_compute_forward_map_custom2

static void ggml_compute_forward_map_custom2(const struct ggml_compute_params* params, const struct ggml_tensor* a,
                                             const struct ggml_tensor* b, struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  struct ggml_map_custom2_op_params* p = (struct ggml_map_custom2_op_params*)dst->op_params;

  p->fun(dst, a, b, params->ith, params->nth, p->userdata);
}

// ggml_compute_forward_map_custom3

static void ggml_compute_forward_map_custom3(const struct ggml_compute_params* params, const struct ggml_tensor* a,
                                             const struct ggml_tensor* b, const struct ggml_tensor* c,
                                             struct ggml_tensor* dst) {
  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  struct ggml_map_custom3_op_params* p = (struct ggml_map_custom3_op_params*)dst->op_params;

  p->fun(dst, a, b, c, params->ith, params->nth, p->userdata);
}

// ggml_compute_forward_cross_entropy_loss

static void ggml_compute_forward_cross_entropy_loss_f32(const struct ggml_compute_params* params,
                                                        const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                        struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous(src0));
  GGML_ASSERT(ggml_is_contiguous(src1));
  GGML_ASSERT(ggml_is_scalar(dst));
  GGML_ASSERT(ggml_are_same_shape(src0, src1));

  const int ith = params->ith;
  const int nth = params->nth;

  float* sums = (float*)params->wdata;

  // TODO: handle transposed/permuted matrices
  const int nc = src0->ne[0];
  const int nr = ggml_nrows(src0);

  GGML_ASSERT(params->wsize >= sizeof(float) * (nth + nth * nc));

  if (params->type == GGML_TASK_INIT) {
    if (ith == 0) {
      memset(sums, 0, sizeof(float) * (nth + nth * nc));
    }
    return;
  }

  if (params->type == GGML_TASK_FINALIZE) {
    if (ith == 0) {
      float* dp = (float*)dst->data;
      ggml_vec_sum_f32(nth, dp, sums);
      dp[0] *= -1.0f / (float)nr;
    }
    return;
  }

  const double eps = 1e-9;

  // rows per thread
  const int dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int ir0 = dr * ith;
  const int ir1 = MIN(ir0 + dr, nr);

  for (int i1 = ir0; i1 < ir1; i1++) {
    float* s0 = (float*)((char*)src0->data + i1 * src0->nb[1]);
    float* s1 = (float*)((char*)src1->data + i1 * src1->nb[1]);
    float* st = ((float*)params->wdata) + nth + ith * nc;

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      // printf("p[%d] = %f\n", i, p[i]);
      assert(!isnan(s0[i]));
      assert(!isnan(s1[i]));
    }
#endif
    // soft_max
    ggml_float sum = 0.0;
    {
      float max = -INFINITY;
      ggml_vec_max_f32(nc, &max, s0);

      uint16_t scvt;
      // UNUSED(scvt);
      for (int i = 0; i < nc; i++) {
        if (s0[i] == -INFINITY) {
          st[i] = 0.0f;
        } else {
#ifndef GGML_CROSS_ENTROPY_EXP_FP16
          const float s = s0[i] - max;
          const float val = expf(s);
#else
          ggml_fp16_t s = GGML_FP32_TO_FP16(s0[i] - max);
          memcpy(&scvt, &s, sizeof(scvt));
          const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt]);
#endif
          sum += (ggml_float)val;
          st[i] = val;
        }
      }

      assert(sum > 0.0);
      // sum = 1.0/sum;
    }
    // avoid log(0) by rescaling from [0..1] to [eps..1]
    sum = (1.0 - eps) / sum;
    ggml_vec_scale_f32(nc, st, sum);
    ggml_vec_add1_f32(nc, st, st, eps);
    ggml_vec_log_f32(nc, st, st);
    ggml_vec_mul_f32(nc, st, st, s1);

    float st_sum = 0;
    ggml_vec_sum_f32(nc, &st_sum, st);
    sums[ith] += st_sum;

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      assert(!isnan(st[i]));
      assert(!isinf(st[i]));
    }
#endif
  }
}

static void ggml_compute_forward_cross_entropy_loss(const struct ggml_compute_params* params,
                                                    const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                    struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_cross_entropy_loss_f32(params, src0, src1, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

// ggml_compute_forward_cross_entropy_loss_back

static void ggml_compute_forward_cross_entropy_loss_back_f32(const struct ggml_compute_params* params,
                                                             const struct ggml_tensor* src0,
                                                             const struct ggml_tensor* src1,
                                                             const struct ggml_tensor* opt0, struct ggml_tensor* dst) {
  GGML_ASSERT(ggml_is_contiguous(dst));
  GGML_ASSERT(ggml_is_contiguous(src0));
  GGML_ASSERT(ggml_is_contiguous(src1));
  GGML_ASSERT(ggml_is_contiguous(opt0));
  GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

  const int64_t ith = params->ith;
  const int64_t nth = params->nth;

  if (params->type == GGML_TASK_INIT || params->type == GGML_TASK_FINALIZE) {
    return;
  }

  const double eps = 1e-9;

  // TODO: handle transposed/permuted matrices
  const int64_t nc = src0->ne[0];
  const int64_t nr = ggml_nrows(src0);

  // rows per thread
  const int64_t dr = (nr + nth - 1) / nth;

  // row range for this thread
  const int64_t ir0 = dr * ith;
  const int64_t ir1 = MIN(ir0 + dr, nr);

  float* d = (float*)opt0->data;

  for (int64_t i1 = ir0; i1 < ir1; i1++) {
    float* ds0 = (float*)((char*)dst->data + i1 * dst->nb[1]);
    float* s0 = (float*)((char*)src0->data + i1 * src0->nb[1]);
    float* s1 = (float*)((char*)src1->data + i1 * src1->nb[1]);

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      // printf("p[%d] = %f\n", i, p[i]);
      assert(!isnan(s0[i]));
      assert(!isnan(s1[i]));
    }
#endif

    // soft_max
    ggml_float sum = 0.0;
    {
      float max = -INFINITY;
      ggml_vec_max_f32(nc, &max, s0);

      uint16_t scvt;
      // UNUSED(scvt);
      for (int i = 0; i < nc; i++) {
        if (s0[i] == -INFINITY) {
          ds0[i] = 0.0f;
        } else {
#ifndef GGML_CROSS_ENTROPY_EXP_FP16
          const float s = s0[i] - max;
          const float val = expf(s);
#else
          ggml_fp16_t s = GGML_FP32_TO_FP16(s0[i] - max);
          memcpy(&scvt, &s, sizeof(scvt));
          const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt]);
#endif
          sum += (ggml_float)val;
          ds0[i] = val;
        }
      }

      assert(sum > 0.0);
      sum = (1.0 - eps) / sum;
    }

    // grad(src0) = (softmax(src0) - src1) * grad(cross_entropy_loss(src0, src1)) / nr
    ggml_vec_scale_f32(nc, ds0, sum);
    ggml_vec_add1_f32(nc, ds0, ds0, eps);
    ggml_vec_sub_f32(nc, ds0, ds0, s1);
    ggml_vec_scale_f32(nc, ds0, d[0] / (float)nr);

#ifndef NDEBUG
    for (int i = 0; i < nc; ++i) {
      assert(!isnan(ds0[i]));
      assert(!isinf(ds0[i]));
    }
#endif
  }
}

static void ggml_compute_forward_cross_entropy_loss_back(const struct ggml_compute_params* params,
                                                         const struct ggml_tensor* src0, const struct ggml_tensor* src1,
                                                         const struct ggml_tensor* opt0, struct ggml_tensor* dst) {
  switch (src0->type) {
    case GGML_TYPE_F32: {
      ggml_compute_forward_cross_entropy_loss_back_f32(params, src0, src1, opt0, dst);
    } break;
    default: {
      GGML_ASSERT(false);
    } break;
  }
}

static void ggml_compute_forward(struct ggml_compute_params* params, struct ggml_tensor* tensor) {
  GGML_ASSERT(params);

#ifdef GGML_USE_CUBLAS
  bool skip_cpu = ggml_cuda_compute_forward(params, tensor);
  if (skip_cpu) {
    return;
  }
  GGML_ASSERT(tensor->src[0] == NULL || tensor->src[0]->backend == GGML_BACKEND_CPU);
  GGML_ASSERT(tensor->src[1] == NULL || tensor->src[1]->backend == GGML_BACKEND_CPU);
#endif  // GGML_USE_CUBLAS

  switch (tensor->op) {
    case GGML_OP_DUP: {
      ggml_compute_forward_dup(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_ADD: {
      ggml_compute_forward_add(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ADD1: {
      ggml_compute_forward_add1(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ACC: {
      ggml_compute_forward_acc(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SUB: {
      ggml_compute_forward_sub(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_MUL: {
      ggml_compute_forward_mul(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_DIV: {
      ggml_compute_forward_div(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SQR: {
      ggml_compute_forward_sqr(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SQRT: {
      ggml_compute_forward_sqrt(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_LOG: {
      ggml_compute_forward_log(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SUM: {
      ggml_compute_forward_sum(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SUM_ROWS: {
      ggml_compute_forward_sum_rows(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_MEAN: {
      ggml_compute_forward_mean(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_ARGMAX: {
      ggml_compute_forward_argmax(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_REPEAT: {
      ggml_compute_forward_repeat(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_REPEAT_BACK: {
      ggml_compute_forward_repeat_back(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_CONCAT: {
      ggml_compute_forward_concat(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SILU_BACK: {
      ggml_compute_forward_silu_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_NORM: {
      ggml_compute_forward_norm(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_RMS_NORM: {
      ggml_compute_forward_rms_norm(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_RMS_NORM_BACK: {
      ggml_compute_forward_rms_norm_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_GROUP_NORM: {
      ggml_compute_forward_group_norm(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_MUL_MAT: {
      ggml_compute_forward_mul_mat(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_OUT_PROD: {
      ggml_compute_forward_out_prod(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SCALE: {
      ggml_compute_forward_scale(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_SET: {
      ggml_compute_forward_set(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_CPY: {
      ggml_compute_forward_cpy(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_CONT: {
      ggml_compute_forward_cont(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_RESHAPE: {
      ggml_compute_forward_reshape(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_VIEW: {
      ggml_compute_forward_view(params, tensor->src[0]);
    } break;
    case GGML_OP_PERMUTE: {
      ggml_compute_forward_permute(params, tensor->src[0]);
    } break;
    case GGML_OP_TRANSPOSE: {
      ggml_compute_forward_transpose(params, tensor->src[0]);
    } break;
    case GGML_OP_GET_ROWS: {
      ggml_compute_forward_get_rows(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_GET_ROWS_BACK: {
      ggml_compute_forward_get_rows_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_DIAG: {
      ggml_compute_forward_diag(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_DIAG_MASK_INF: {
      ggml_compute_forward_diag_mask_inf(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_DIAG_MASK_ZERO: {
      ggml_compute_forward_diag_mask_zero(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SOFT_MAX: {
      ggml_compute_forward_soft_max(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_SOFT_MAX_BACK: {
      ggml_compute_forward_soft_max_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ROPE: {
      ggml_compute_forward_rope(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ROPE_BACK: {
      ggml_compute_forward_rope_back(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_ALIBI: {
      ggml_compute_forward_alibi(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_CLAMP: {
      ggml_compute_forward_clamp(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_CONV_1D: {
      ggml_compute_forward_conv_1d(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_CONV_2D: {
      ggml_compute_forward_conv_2d(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_CONV_TRANSPOSE_2D: {
      ggml_compute_forward_conv_transpose_2d(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_POOL_1D: {
      ggml_compute_forward_pool_1d(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_POOL_2D: {
      ggml_compute_forward_pool_2d(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_UPSCALE: {
      ggml_compute_forward_upscale(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_FLASH_ATTN: {
      const int32_t t = ggml_get_op_params_i32(tensor, 0);
      GGML_ASSERT(t == 0 || t == 1);
      const bool masked = t != 0;
      ggml_compute_forward_flash_attn(params, tensor->src[0], tensor->src[1], tensor->src[2], masked, tensor);
    } break;
    case GGML_OP_FLASH_FF: {
      ggml_compute_forward_flash_ff(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3],
                                    tensor->src[4], tensor);
    } break;
    case GGML_OP_FLASH_ATTN_BACK: {
      int32_t t = ggml_get_op_params_i32(tensor, 0);
      GGML_ASSERT(t == 0 || t == 1);
      bool masked = t != 0;
      ggml_compute_forward_flash_attn_back(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor->src[3],
                                           masked, tensor);
    } break;
    case GGML_OP_WIN_PART: {
      ggml_compute_forward_win_part(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_WIN_UNPART: {
      ggml_compute_forward_win_unpart(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_UNARY: {
      ggml_compute_forward_unary(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_GET_REL_POS: {
      ggml_compute_forward_get_rel_pos(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_ADD_REL_POS: {
      ggml_compute_forward_add_rel_pos(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_MAP_UNARY: {
      ggml_unary_op_f32_t fun;
      memcpy(&fun, tensor->op_params, sizeof(fun));
      ggml_compute_forward_map_unary(params, tensor->src[0], tensor, fun);
    } break;
    case GGML_OP_MAP_BINARY: {
      ggml_binary_op_f32_t fun;
      memcpy(&fun, tensor->op_params, sizeof(fun));
      ggml_compute_forward_map_binary(params, tensor->src[0], tensor->src[1], tensor, fun);
    } break;
    case GGML_OP_MAP_CUSTOM1_F32: {
      ggml_custom1_op_f32_t fun;
      memcpy(&fun, tensor->op_params, sizeof(fun));
      ggml_compute_forward_map_custom1_f32(params, tensor->src[0], tensor, fun);
    } break;
    case GGML_OP_MAP_CUSTOM2_F32: {
      ggml_custom2_op_f32_t fun;
      memcpy(&fun, tensor->op_params, sizeof(fun));
      ggml_compute_forward_map_custom2_f32(params, tensor->src[0], tensor->src[1], tensor, fun);
    } break;
    case GGML_OP_MAP_CUSTOM3_F32: {
      ggml_custom3_op_f32_t fun;
      memcpy(&fun, tensor->op_params, sizeof(fun));
      ggml_compute_forward_map_custom3_f32(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor, fun);
    } break;
    case GGML_OP_MAP_CUSTOM1: {
      ggml_compute_forward_map_custom1(params, tensor->src[0], tensor);
    } break;
    case GGML_OP_MAP_CUSTOM2: {
      ggml_compute_forward_map_custom2(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_MAP_CUSTOM3: {
      ggml_compute_forward_map_custom3(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_CROSS_ENTROPY_LOSS: {
      ggml_compute_forward_cross_entropy_loss(params, tensor->src[0], tensor->src[1], tensor);
    } break;
    case GGML_OP_CROSS_ENTROPY_LOSS_BACK: {
      ggml_compute_forward_cross_entropy_loss_back(params, tensor->src[0], tensor->src[1], tensor->src[2], tensor);
    } break;
    case GGML_OP_NONE: {
      // nop
    } break;
    case GGML_OP_COUNT: {
      GGML_ASSERT(false);
    } break;
  }
}

struct ggml_compute_state_shared {
  const struct ggml_cgraph* cgraph;
  const struct ggml_cplan* cplan;

  int64_t perf_node_start_cycles;
  int64_t perf_node_start_time_us;

  const int n_threads;

  // synchronization primitives
  std::atomic_int n_active;  // num active threads
  std::atomic_int node_n;    // active graph node

  bool (*abort_callback)(void* data);  // abort ggml_graph_compute when true
  void* abort_callback_data;
};

struct ggml_compute_state {
  ggml_thread_t thrd;
  int ith;
  struct ggml_compute_state_shared* shared;
};

static void ggml_graph_compute_perf_stats_node(struct ggml_tensor* node, const struct ggml_compute_state_shared* st) {
  int64_t cycles_cur = ggml_perf_cycles() - st->perf_node_start_cycles;
  int64_t time_us_cur = ggml_perf_time_us() - st->perf_node_start_time_us;

  node->perf_runs++;
  node->perf_cycles += cycles_cur;
  node->perf_time_us += time_us_cur;
}

static thread_ret_t ggml_graph_compute_thread(void* data) {
  struct ggml_compute_state* state = (struct ggml_compute_state*)data;

  const struct ggml_cgraph* cgraph = state->shared->cgraph;
  const struct ggml_cplan* cplan = state->shared->cplan;

  const int* n_tasks_arr = cplan->n_tasks;
  const int n_threads = state->shared->n_threads;

  set_numa_thread_affinity(state->ith, n_threads);

  int node_n = -1;

  while (true) {
    if (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
      state->shared->node_n += 1;
      return (thread_ret_t)GGML_EXIT_ABORTED;
    }
    if (atomic_fetch_sub(&state->shared->n_active, 1) == 1) {
      // all other threads are finished and spinning
      // do finalize and init here so we don't have synchronize again
      struct ggml_compute_params params = {
          /*.type  =*/GGML_TASK_FINALIZE,
          /*.ith   =*/0,
          /*.nth   =*/0,
          /*.wsize =*/cplan->work_size,
          /*.wdata =*/cplan->work_data,
      };

      if (node_n != -1) {
        /* FINALIZE */
        struct ggml_tensor* node = state->shared->cgraph->nodes[node_n];
        if (GGML_OP_HAS_FINALIZE[node->op]) {
          params.nth = n_tasks_arr[node_n];
          ggml_compute_forward(&params, node);
        }
        ggml_graph_compute_perf_stats_node(node, state->shared);
      }

      // distribute new work or execute it direct if 1T
      while (++node_n < cgraph->n_nodes) {
        struct ggml_tensor* node = cgraph->nodes[node_n];
        GGML_PRINT_DEBUG_5("%s: %s (%d/%d)\n", __func__, node->name, node_n, cgraph->n_nodes);

        const int n_tasks = n_tasks_arr[node_n];

        state->shared->perf_node_start_cycles = ggml_perf_cycles();
        state->shared->perf_node_start_time_us = ggml_perf_time_us();

        params.nth = n_tasks;

        /* INIT */
        if (GGML_OP_HAS_INIT[node->op]) {
          params.type = GGML_TASK_INIT;
          ggml_compute_forward(&params, node);
        }

        if (n_tasks == 1) {
          // TODO: maybe push node_n to the atomic but if other threads see n_tasks is 1,
          // they do something more efficient than spinning (?)
          params.type = GGML_TASK_COMPUTE;
          std::string node_name = node->name;
          printf("%s: compute node: %-40s (%d/%d)\n", __func__, node_name.c_str(), node_n, cgraph->n_nodes);
          ggml_compute_forward(&params, node);

          if (GGML_OP_HAS_FINALIZE[node->op]) {
            params.type = GGML_TASK_FINALIZE;
            ggml_compute_forward(&params, node);
          }

          ggml_graph_compute_perf_stats_node(node, state->shared);
        } else {
          break;
        }

        if (cplan->abort_callback && cplan->abort_callback(cplan->abort_callback_data)) {
          break;
        }
      }

      atomic_store(&state->shared->n_active, n_threads);
      atomic_store(&state->shared->node_n, node_n);
    } else {
      // wait for other threads to finish
      const int last = node_n;
      while (true) {
        // TODO: this sched_yield can have significant impact on the performance - either positive or negative
        //       depending on the workload and the operating system.
        //       since it is not clear what is the best approach, it should potentially become user-configurable
        //       ref: https://github.com/ggerganov/ggml/issues/291
#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
        sched_yield();
#endif

        node_n = atomic_load(&state->shared->node_n);
        if (node_n != last) break;
      };
    }

    // check if we should stop
    if (node_n >= cgraph->n_nodes) break;

    /* COMPUTE */
    struct ggml_tensor* node = cgraph->nodes[node_n];
    const int n_tasks = n_tasks_arr[node_n];

    struct ggml_compute_params params = {
        /*.type  =*/GGML_TASK_COMPUTE,
        /*.ith   =*/state->ith,
        /*.nth   =*/n_tasks,
        /*.wsize =*/cplan->work_size,
        /*.wdata =*/cplan->work_data,
    };

    if (state->ith < n_tasks) {
      ggml_compute_forward(&params, node);
    }
  }

  return GGML_EXIT_SUCCESS;
}

int ggml_graph_compute(struct ggml_cgraph* cgraph, struct ggml_cplan* cplan) {
  {
    GGML_ASSERT(cplan);
    GGML_ASSERT(cplan->n_threads > 0);

    if (cplan->work_size > 0) {
      GGML_ASSERT(cplan->work_data);
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
      if (cgraph->nodes[i]->op != GGML_OP_NONE) {
        GGML_ASSERT(cplan->n_tasks[i] > 0);
      }
    }
  }

  const int n_threads = cplan->n_threads;

  struct ggml_compute_state_shared state_shared = {
      .cgraph = cgraph,
      .cplan = cplan,
      .perf_node_start_cycles = 0,
      .perf_node_start_time_us = 0,
      .n_threads = n_threads,
      .abort_callback = NULL,
      .abort_callback_data = NULL,
  };
  state_shared.node_n = -1;
  state_shared.n_active = n_threads;

  struct ggml_compute_state* workers =
      (struct ggml_compute_state*)alloca(sizeof(struct ggml_compute_state) * n_threads);

  // create thread pool
  if (n_threads > 1) {
    for (int j = 1; j < n_threads; ++j) {
      workers[j] = (struct ggml_compute_state){
          .thrd = 0,
          .ith = j,
          .shared = &state_shared,
      };

      const int rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
      GGML_ASSERT(rc == 0);
      // UNUSED(rc);
    }
  }

  workers[0].ith = 0;
  workers[0].shared = &state_shared;

  const int64_t perf_start_cycles = ggml_perf_cycles();
  const int64_t perf_start_time_us = ggml_perf_time_us();

  // this is a work thread too
  int compute_status = (size_t)ggml_graph_compute_thread(&workers[0]);

  // don't leave affinity set on the main thread
  clear_numa_thread_affinity();

  // join or kill thread pool
  if (n_threads > 1) {
    for (int j = 1; j < n_threads; j++) {
      const int rc = ggml_thread_join(workers[j].thrd, NULL);
      GGML_ASSERT(rc == 0);
    }
  }

  // performance stats (graph)
  {
    int64_t perf_cycles_cur = ggml_perf_cycles() - perf_start_cycles;
    int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

    cgraph->perf_runs++;
    cgraph->perf_cycles += perf_cycles_cur;
    cgraph->perf_time_us += perf_time_us_cur;

    GGML_PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n", __func__, cgraph->perf_runs,
                     (double)perf_cycles_cur / (double)ggml_cycles_per_ms(),
                     (double)cgraph->perf_cycles / (double)ggml_cycles_per_ms() / (double)cgraph->perf_runs,
                     (double)perf_time_us_cur / 1000.0, (double)cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
  }

  return compute_status;
}

#define __GGML_ALLOCATOR_FUNCTIONS__
// =======================================================================
// GGML allocator functions
// =======================================================================
void ggml_allocator_reset(struct ggml_allocator* allocator) {
  allocator->n_free_blocks = 1;
  size_t align_offset = aligned_offset(allocator->data, 0, allocator->alignment);
  allocator->free_blocks[0].addr = (char*)allocator->data + align_offset;
  allocator->free_blocks[0].size = allocator->size - align_offset;
}

struct ggml_allocator* ggml_allocator_new(void* data, size_t size, size_t alignment) {
  struct ggml_allocator* allocator =
      (struct ggml_allocator*)malloc(sizeof(struct ggml_allocator) /* + n_free_blocks * sizeof(struct free_block) */);

  *allocator = (struct ggml_allocator){
      /*.data          = */ data,
      /*.size          = */ size,
      /*.alignment     = */ alignment,
      /*.n_free_blocks = */ 0,
      /*.free_blocks   = */ {{0}},
      /*.hash_table    = */ {{0}},
      /*.max_size      = */ 0,
      /*.measure       = */ false,
      /*.parse_seq     = */ {0},
      /*.parse_seq_len = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
      /*.allocated_tensors = */ {0},
#endif
  };

  ggml_allocator_reset(allocator);

  return allocator;
}

void ggml_allocator_alloc(struct ggml_allocator* allocator, struct ggml_tensor* tensor) {
#ifdef GGML_ALLOCATOR_DEBUG
  GGML_ASSERT(!tensor->view_src != NULL);  // views generally get data pointer from one of their sources
  GGML_ASSERT(tensor->data == NULL);       // avoid allocating tensor which already has memory allocated
#endif
  size_t size = ggml_nbytes(tensor);
  size = aligned_offset(NULL, size, allocator->alignment);

  printf("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

  size_t max_avail = 0;

  // find the best fitting free block besides the last block
  int best_fit_block = -1;
  size_t best_fit_size = SIZE_MAX;
  for (int i = 0; i < allocator->n_free_blocks - 1; i++) {
    struct free_block* block = &allocator->free_blocks[i];
    max_avail = MAX(max_avail, block->size);
    if (block->size >= size && block->size <= best_fit_size) {
      best_fit_block = i;
      best_fit_size = block->size;
    }
  }

  printf("block %d\n", best_fit_block);

  if (best_fit_block == -1) {
    // the last block is our last resort
    struct free_block* block = &allocator->free_blocks[allocator->n_free_blocks - 1];
    max_avail = MAX(max_avail, block->size);
    if (block->size >= size) {
      best_fit_block = allocator->n_free_blocks - 1;
    } else {
      fprintf(stderr, "%s: not enough space in the buffer (needed %zu, largest block available %zu)\n", __func__, size,
              max_avail);
      GGML_ASSERT(!"not enough space in the buffer");
      return;
    }
  }
  struct free_block* block = &allocator->free_blocks[best_fit_block];
  void* addr = block->addr;
  block->addr = (char*)block->addr + size;
  block->size -= size;
  if (block->size == 0) {
    // remove block if empty
    allocator->n_free_blocks--;
    for (int j = best_fit_block; j < allocator->n_free_blocks; j++) {
      allocator->free_blocks[j] = allocator->free_blocks[j + 1];
    }
  }

  tensor->data = addr;
  printf("%s: allocated data at %p\n", __func__, tensor->data);

#ifdef GGML_ALLOCATOR_DEBUG
  add_allocated_tensor(allocator, tensor);
  size_t cur_max = (char*)addr - (char*)allocator->data + size;
  if (cur_max > allocator->max_size) {
    printf("max_size = %.2f MB: tensors: ", cur_max / 1024.0 / 1024.0);
    for (int i = 0; i < 1024; i++) {
      if (allocator->allocated_tensors[i]) {
        printf("%s (%.2f MB) ", allocator->allocated_tensors[i]->name,
               ggml_nbytes(allocator->allocated_tensors[i]) / 1024.0 / 1024.0);
      }
    }
    printf("\n");
  }
#endif

  allocator->max_size = MAX(allocator->max_size, (char*)addr - (char*)allocator->data + size);
  printf("\n");
}

// check if a tensor is allocated by this buffer
static bool ggml_allocator_is_own(struct ggml_allocator* allocator, const struct ggml_tensor* tensor) {
  void* ptr = tensor->data;
  return ptr >= allocator->data && (char*)ptr < (char*)allocator->data + allocator->max_size;
}

static void allocate_node(struct ggml_allocator* allocator, struct ggml_tensor* node) {
  struct hash_node* ht = allocator->hash_table;
  if (node->data == NULL) {
    if (node->view_src != NULL) {
      assert(node->view_src->data != NULL);
      node->data = (char*)node->view_src->data + node->view_offs;
    } else {
      // see if we can reuse a parent's buffer (inplace)
      if (ggml_op_can_inplace(node->op)) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
          struct ggml_tensor* parent = node->src[i];
          if (parent == NULL) {
            break;
          }

          // if the node's data is external, then we cannot re-use it
          if (ggml_allocator_is_own(allocator, parent) == false) {
            printf("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
            continue;
          }

          struct hash_node* p_hn = hash_get(ht, parent);
          if (parent->data != NULL && p_hn->n_children == 1 && p_hn->n_views == 0 &&
              ggml_are_same_layout(node, parent)) {
            if (parent->view_src != NULL) {
              struct ggml_tensor* view_src = parent->view_src;
              struct hash_node* view_src_hn = hash_get(ht, view_src);
              if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
                // TODO: the offset of the view parent must be kept to ensure that the op doesn't overwrite
                // the parent's data that it will need later (same layout requirement). the problem is that then
                // we cannot free the tensor because the original address of the allocation is lost.
                // adding a view_src pointer to the tensor would solve this and simplify the code dealing with views
                // for now, we only reuse the parent's data if the offset is zero (view_src->data == parent->data)
                printf("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
                node->data = parent->data;
                return;
              }
            } else {
              printf("reusing parent %s for %s\n", parent->name, node->name);
              node->data = parent->data;
              return;
            }
          }
        }
      }
      ggml_allocator_alloc(allocator, node);
    }
  }
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
static void ggml_allocator_free_tensor(struct ggml_allocator* allocator, struct ggml_tensor* tensor) {
  void* ptr = tensor->data;

  if (ggml_allocator_is_own(allocator, tensor) == false) {
    // the tensor was not allocated in this buffer
    // this can happen because the graph allocator will try to free weights and other tensors from different buffers
    // the easiest way to deal with this is just to ignore it
    return;
  }

  size_t size = ggml_nbytes(tensor);
  size = aligned_offset(NULL, size, allocator->alignment);
  printf("%s: freeing %s at %p (%zu bytes) - n_free_blocks = %d\n", __func__, tensor->name, ptr, size,
         allocator->n_free_blocks);
  printf("%s: allocator->data = %p allocator->data+allocator->size = %p allocator->data+allocator->max_size = %p\n",
         __func__, allocator->data, (char*)allocator->data + allocator->size,
         (char*)allocator->data + allocator->max_size);

#ifdef GGML_ALLOCATOR_DEBUG
  remove_allocated_tensor(allocator, tensor);
#endif

  // see if we can merge with an existing block
  for (int i = 0; i < allocator->n_free_blocks; i++) {
    struct free_block* block = &allocator->free_blocks[i];
    // check if ptr is at the end of the block
    if ((char*)block->addr + block->size == ptr) {
      block->size += size;
      // check if we can merge with the next block
      if (i < allocator->n_free_blocks - 1 && (char*)block->addr + block->size == allocator->free_blocks[i + 1].addr) {
        block->size += allocator->free_blocks[i + 1].size;
        allocator->n_free_blocks--;
        for (int j = i + 1; j < allocator->n_free_blocks; j++) {
          allocator->free_blocks[j] = allocator->free_blocks[j + 1];
        }
      }
      return;
    }
    // check if ptr is at the beginning of the block
    if ((char*)ptr + size == block->addr) {
      block->addr = ptr;
      block->size += size;
      // check if we can merge with the previous block
      if (i > 0 && (char*)allocator->free_blocks[i - 1].addr + allocator->free_blocks[i - 1].size == block->addr) {
        allocator->free_blocks[i - 1].size += block->size;
        allocator->n_free_blocks--;
        for (int j = i; j < allocator->n_free_blocks; j++) {
          allocator->free_blocks[j] = allocator->free_blocks[j + 1];
        }
      }
      return;
    }
  }
  // otherwise, add a new block
  GGML_ASSERT(allocator->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
  // insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
  int insert_pos = 0;
  while (insert_pos < allocator->n_free_blocks && allocator->free_blocks[insert_pos].addr < ptr) {
    insert_pos++;
  }
  // shift all blocks from insert_pos onward to make room for the new block
  for (int i = allocator->n_free_blocks; i > insert_pos; i--) {
    allocator->free_blocks[i] = allocator->free_blocks[i - 1];
  }
  // insert the new block
  allocator->free_blocks[insert_pos].addr = ptr;
  allocator->free_blocks[insert_pos].size = size;
  allocator->n_free_blocks++;
}

static size_t ggml_allocator_alloc_graph_tensors_n(struct ggml_allocator* allocator, struct ggml_cgraph** graphs,
                                                   int n_graphs, struct ggml_tensor*** inputs,
                                                   struct ggml_tensor*** outputs) {
  // reset hash table
  struct hash_node* ht = allocator->hash_table;
  memset(ht, 0, sizeof(struct hash_node) * GGML_GRAPH_HASHTABLE_SIZE);

  // count number of children and views
  for (int g = 0; g < n_graphs; g++) {
    struct ggml_cgraph* gf = graphs[g];
    for (int i = 0; i < gf->n_nodes; i++) {
      struct ggml_tensor* node = gf->nodes[i];

      if (node->view_src != NULL) {
        struct ggml_tensor* view_src = node->view_src;
        hash_get(ht, view_src)->n_views += 1;
      }

      for (int j = 0; j < GGML_MAX_SRC; j++) {
        struct ggml_tensor* parent = node->src[j];
        if (parent == NULL) {
          break;
        }
        hash_get(ht, parent)->n_children += 1;
      }
    }
  }

  // allocate tensors
  for (int g = 0; g < n_graphs; g++) {
    struct ggml_cgraph* gf = graphs[g];
    printf("####### graph %d/%d\n", g, n_graphs);
    // graph inputs are allocated first to ensure that they are not overwritten by each other
    if (inputs != NULL && inputs[g] != NULL) {
      for (int i = 0; inputs[g][i] != NULL; i++) {
        struct ggml_tensor* input = inputs[g][i];
        printf("input: %s\n", input->name);
        allocate_node(allocator, input);
      }
    }
    // if we have parse_seq then we allocate nodes following the list, and we only free nodes at barriers
    int last_barrier_pos = 0;
    int n_nodes = allocator->parse_seq_len ? allocator->parse_seq_len : gf->n_nodes;

    for (int ind = 0; ind < n_nodes; ind++) {
      // allocate a node if there is no parse_seq or this is not a barrier
      if ((allocator->parse_seq_len == 0) || allocator->parse_seq[ind] != -1) {
        int i = allocator->parse_seq_len ? allocator->parse_seq[ind] : ind;
        struct ggml_tensor* node = gf->nodes[i];

        // allocate parents (leafs)
        for (int j = 0; j < GGML_MAX_SRC; j++) {
          struct ggml_tensor* parent = node->src[j];
          if (parent == NULL) {
            break;
          }
          allocate_node(allocator, parent);
        }

        // allocate node
        allocate_node(allocator, node);

        printf("exec: %s (%s) <= ", GGML_OP_NAMES[node->op], node->name);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
          struct ggml_tensor* parent = node->src[j];
          if (parent == NULL) {
            break;
          }
          printf("%s", parent->name);
          if (j < GGML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
            printf(", ");
          }
        }
        printf("\n");
      }

      // update parents
      // update immediately if there is no parse_seq
      // update only at barriers if there is parse_seq
      if ((allocator->parse_seq_len == 0) || allocator->parse_seq[ind] == -1) {
        int update_start = allocator->parse_seq_len ? last_barrier_pos : ind;
        int update_end = allocator->parse_seq_len ? ind : ind + 1;
        for (int i = update_start; i < update_end; i++) {
          int node_i = allocator->parse_seq_len ? allocator->parse_seq[i] : i;
          struct ggml_tensor* node = gf->nodes[node_i];

          for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor* parent = node->src[j];
            if (parent == NULL) {
              break;
            }
            struct hash_node* p_hn = hash_get(ht, parent);
            p_hn->n_children -= 1;

            // AT_PRINTF("parent %s: %d children, %d views\n", parent->name, parent->n_children, parent->n_views);

            if (p_hn->n_children == 0 && p_hn->n_views == 0) {
              if (parent->view_src != NULL) {
                struct ggml_tensor* view_src = parent->view_src;
                struct hash_node* view_src_hn = hash_get(ht, view_src);
                view_src_hn->n_views -= 1;
                printf("view_src %s: %d children, %d views\n", view_src->name, view_src_hn->n_children,
                       view_src_hn->n_views);
                if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src->data != node->data) {
                  ggml_allocator_free_tensor(allocator, view_src);
                }
              } else {
                if (parent->data != node->data) {
                  ggml_allocator_free_tensor(allocator, parent);
                }
              }
            }
          }
        }
        printf("\n");
        if (allocator->parse_seq_len) {
          last_barrier_pos = ind + 1;
        }
      }
    }
    // free graph outputs here that wouldn't be freed otherwise because they have no children
    if (outputs != NULL && outputs[g] != NULL) {
      for (int i = 0; outputs[g][i] != NULL; i++) {
        struct ggml_tensor* output = outputs[g][i];
        printf("output: %s\n", output->name);
        ggml_allocator_free_tensor(allocator, output);
      }
    }
  }

  return allocator->max_size;
}

size_t ggml_allocator_alloc_graph(struct ggml_allocator* allocator, struct ggml_cgraph* graph) {
  return ggml_allocator_alloc_graph_tensors_n(allocator, &graph, 1, NULL, NULL);
}

// OS specific functions to allocate and free uncommitted virtual memory
static void* alloc_vmem(size_t size) {
#if defined(_POSIX_MAPPED_FILES)
  void* ptr = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (ptr == MAP_FAILED) {
    return NULL;
  }
  return ptr;
#else
  // use a fixed address for other platforms
  uintptr_t base_addr = (uintptr_t)-size - 0x100;
  return (void*)base_addr;
#endif
}

static void free_vmem(void* base_addr, size_t size) {
#if defined(_POSIX_MAPPED_FILES)
  munmap(base_addr, size);
#else
  // nothing to do
  (void)base_addr;
  (void)size;
#endif
}

void ggml_allocator_free(struct ggml_allocator* allocator) {
  if (allocator->measure) {
    free_vmem(allocator->data, allocator->size);
  }
  free(allocator);
}

// allocate uncommitted virtual memory to measure the size of the graph
static void alloc_measure_vmem(void** base_addr, size_t* size) {
  // 128GB for 64-bit, 1GB for 32-bit
  *size = sizeof(void*) == 4 ? 1ULL << 30 : 1ULL << 37;
  do {
    *base_addr = alloc_vmem(*size);
    if (*base_addr != NULL) {
      printf("allocated %.2f GB of virtual memory for measure buffer at %p\n", *size / 1024.0 / 1024.0 / 1024.0,
             *base_addr);
      return;
    }
    // try again with half the size
    *size /= 2;
  } while (*size > 0);

  GGML_ASSERT(!"failed to allocate virtual memory for measure buffer");
}

struct ggml_allocator* ggml_allocator_new_measure(size_t alignment) {
  struct ggml_allocator* allocator =
      (struct ggml_allocator*)malloc(sizeof(struct ggml_allocator) /* + n_free_blocks * sizeof(struct free_block) */);

  void* base_addr;
  size_t size;

  alloc_measure_vmem(&base_addr, &size);

  *allocator = (struct ggml_allocator){
      /*.data          = */ base_addr,
      /*.size          = */ size,
      /*.alignment     = */ alignment,
      /*.n_free_blocks = */ 0,
      /*.free_blocks   = */ {{0}},
      /*.hash_table    = */ {{0}},
      /*.max_size      = */ 0,
      /*.measure       = */ true,
      /*.parse_seq     = */ {0},
      /*.parse_seq_len = */ 0,
#ifdef GGML_ALLOCATOR_DEBUG
      /*.allocated_tensors = */ {0},
#endif
  };

  ggml_allocator_reset(allocator);

  return allocator;
}

#define __LLAMA_MODEL_DSA__
// =======================================================================
// LLaMa Model
// =======================================================================
LLAMA_ATTRIBUTE_FORMAT(2, 3)
static void llama_log_internal(ggml_log_level level, const char* format, ...);
static void llama_log_callback_default(ggml_log_level level, const char* text, void* user_data);

#define LLAMA_LOG_INFO(...) llama_log_internal(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define LLAMA_LOG_WARN(...) llama_log_internal(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define LLAMA_LOG_ERROR(...) llama_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)

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

void llama_backend_init(bool numa) {
  ggml_time_init();

  // needed to initialize f16 tables
  {
    struct ggml_init_params params = {0, NULL, false};
    struct ggml_context* ctx = ggml_init(params);
    ggml_free(ctx);
  }

  if (numa) {
    ggml_numa_init();
  }
}

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

struct llama_cparams {
  uint32_t n_ctx;  // context size used during inference
  uint32_t n_batch;
  uint32_t n_threads;        // number of threads to use for generation
  uint32_t n_threads_batch;  // number of threads to use for batch processing

  float rope_freq_base;
  float rope_freq_scale;

  bool mul_mat_q;
};

struct llama_kv_cell {
  llama_pos pos = -1;
  llama_pos delta = 0;

  std::set<llama_seq_id> seq_id;

  bool has_seq_id(const llama_seq_id& id) const { return seq_id.find(id) != seq_id.end(); }
};

// ring-buffer of cached KV data
struct llama_kv_cache {
  bool has_shift = false;

  uint32_t head = 0;
  uint32_t size = 0;

  // computed before each graph build
  uint32_t n = 0;

  std::vector<llama_kv_cell> cells;

  struct ggml_tensor* k = NULL;
  struct ggml_tensor* v = NULL;

  struct ggml_context* ctx = NULL;

  llama_buffer buf;

  ~llama_kv_cache() {
    if (ctx) {
      ggml_free(ctx);
    }

#ifdef GGML_USE_CUBLAS
    ggml_cuda_free_data(k);
    ggml_cuda_free_data(v);
#endif  // GGML_USE_CUBLAS
  }
};

struct llama_context {
  llama_context(const llama_model* model) : model(model), t_start_us(model->t_start_us), t_load_us(model->t_load_us) {}
  ~llama_context() {
    if (allocator) {
      ggml_allocator_free(allocator);
    }
  }

  llama_cparams cparams;

  const llama_model* model;

  // key + value cache for the self attention
  struct llama_kv_cache kv_self;

  std::mt19937 rng;

  bool has_evaluated_once = false;

  int64_t t_start_us;
  int64_t t_load_us;
  int64_t t_sample_us = 0;
  int64_t t_p_eval_us = 0;
  int64_t t_eval_us = 0;

  int32_t n_sample = 0;  // number of tokens sampled
  int32_t n_p_eval = 0;  // number of tokens in eval calls for the prompt (with batch size > 1)
  int32_t n_eval = 0;    // number of eval calls

  // decode output (2-dimensional array: [n_tokens][n_vocab])
  std::vector<float> logits;
  bool logits_all = false;

  // input embedding (1-dimensional array: [n_embd])
  std::vector<float> embedding;

  // reusable buffer for `struct ggml_graph_plan.work_data`
  std::vector<uint8_t> work_buffer;

  // memory buffers used to evaluate the model
  llama_buffer buf_compute;

  llama_buffer buf_alloc;
  ggml_allocator* allocator = NULL;
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

static std::map<std::string, llm_arch> LLM_ARCH_NAMES = {
    {"llama", LLM_ARCH_LLAMA},       {"falcon", LLM_ARCH_FALCON},       {"gpt2", LLM_ARCH_GPT2},
    {"gptj", LLM_ARCH_GPTJ},         {"gptneox", LLM_ARCH_GPTNEOX},     {"mpt", LLM_ARCH_MPT},
    {"baichuan", LLM_ARCH_BAICHUAN}, {"starcoder", LLM_ARCH_STARCODER},
};

static bool llama_kv_cache_init(const struct llama_hparams& hparams, struct llama_kv_cache& cache, ggml_type wtype,
                                uint32_t n_ctx, int n_gpu_layers) {
  const uint32_t n_embd = hparams.n_embd_gqa();
  const uint32_t n_layer = hparams.n_layer;

  const int64_t n_mem = n_layer * n_ctx;
  const int64_t n_elements = n_embd * n_mem;

  cache.has_shift = false;

  cache.head = 0;
  cache.size = n_ctx;

  cache.cells.clear();
  cache.cells.resize(n_ctx);

  cache.buf.resize(2u * n_elements * type_traits[wtype].type_size + 2u * MB);

  struct ggml_init_params params;
  params.mem_size = cache.buf.size;
  params.mem_buffer = cache.buf.data;
  params.no_alloc = false;

  cache.ctx = ggml_init(params);

  if (!cache.ctx) {
    LLAMA_LOG_ERROR("%s: failed to allocate memory for kv cache\n", __func__);
    return false;
  }

  cache.k = ggml_new_tensor(cache.ctx, wtype, 1, new int64_t[1]{n_elements});
  cache.v = ggml_new_tensor(cache.ctx, wtype, 1, new int64_t[1]{n_elements});
  ggml_set_name(cache.k, "cache_k");
  ggml_set_name(cache.v, "cache_v");

  (void)n_gpu_layers;

  if (ggml_use_cublas()) {
    size_t vram_kv_cache = 0;

    if (n_gpu_layers > (int)n_layer + 1) {
      ggml_cuda_assign_buffers_no_scratch(cache.v);
      LLAMA_LOG_INFO("%s: offloading v cache to GPU\n", __func__);
      vram_kv_cache += ggml_nbytes(cache.v);
    }
    if (n_gpu_layers > (int)n_layer + 2) {
      ggml_cuda_assign_buffers_no_scratch(cache.k);
      LLAMA_LOG_INFO("%s: offloading k cache to GPU\n", __func__);
      vram_kv_cache += ggml_nbytes(cache.k);
    }
    if (vram_kv_cache > 0) {
      LLAMA_LOG_INFO("%s: VRAM kv self = %.2f MB\n", __func__, vram_kv_cache / 1024.0 / 1024.0);
    }
  }

  return true;
}

struct llama_context_params llama_context_default_params() {
  struct llama_context_params result = {
      /*.seed                        =*/LLAMA_DEFAULT_SEED,
      /*.n_ctx                       =*/512,
      /*.n_batch                     =*/512,
      /*.n_threads                   =*/GGML_DEFAULT_N_THREADS,  // TODO: better default
      /*.n_threads_batch             =*/GGML_DEFAULT_N_THREADS,
      /*.rope_freq_base              =*/0.0f,
      /*.rope_freq_scale             =*/0.0f,
      /*.mul_mat_q                   =*/true,
      /*.f16_kv                      =*/true,
      /*.logits_all                  =*/false,
      /*.embedding                   =*/false,
  };

  return result;
}

// find an empty slot of size "n_tokens" in the cache
// updates the cache head
static bool llama_kv_cache_find_slot(struct llama_kv_cache& cache, const struct llama_batch& batch) {
  const uint32_t n_ctx = cache.size;
  const uint32_t n_tokens = batch.n_tokens;

  if (n_tokens > n_ctx) {
    LLAMA_LOG_ERROR("%s: n_tokens=%d > n_ctx=%d\n", __func__, n_tokens, n_ctx);
    return false;
  }

  uint32_t n_tested = 0;

  while (true) {
    if (cache.head + n_tokens > n_ctx) {
      cache.head = 0;
      n_tested += n_ctx - cache.head;
      continue;
    }

    bool found = true;
    for (uint32_t i = 0; i < n_tokens; i++) {
      if (cache.cells[cache.head + i].pos >= 0) {
        found = false;
        cache.head += i + 1;
        n_tested += i + 1;
        break;
      }
    }

    if (found) {
      break;
    }

    if (n_tested >= n_ctx) {
      // LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
      return false;
    }
  }

  for (uint32_t i = 0; i < n_tokens; i++) {
    cache.cells[cache.head + i].pos = batch.pos[i];
    cache.cells[cache.head + i].seq_id.insert(batch.seq_id[i]);
  }

  return true;
}

// find how many cells are currently in use
static int32_t llama_kv_cache_cell_max(const struct llama_kv_cache& cache) {
  for (uint32_t i = cache.size - 1; i > 0; --i) {
    if (cache.cells[i].pos >= 0 && !cache.cells[i].seq_id.empty()) {
      return i + 1;
    }
  }

  return 0;
}

static void llama_kv_cache_tokens_rm(struct llama_kv_cache& cache, int32_t c0, int32_t c1) {
  if (c0 < 0) c0 = 0;
  if (c1 < 0) c1 = cache.size;

  for (int32_t i = c0; i < c1; ++i) {
    cache.cells[i].pos = -1;
    cache.cells[i].seq_id.clear();
  }
}

static void llama_nop(struct ggml_tensor* tensor) {  // don't offload by default
  (void)tensor;
}

void llama_reset_timings(struct llama_context* ctx) {
  ctx->t_start_us = ggml_time_us();
  ctx->t_sample_us = ctx->n_sample = 0;
  ctx->t_eval_us = ctx->n_eval = 0;
  ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

struct ggml_cgraph* ggml_new_graph(struct ggml_context* ctx) {
  struct ggml_object* obj = ggml_new_object(ctx, GGML_OBJECT_GRAPH, GGML_GRAPH_SIZE);
  struct ggml_cgraph* cgraph = (struct ggml_cgraph*)((char*)ctx->mem_buffer + obj->offs);

  *cgraph = (struct ggml_cgraph){
      /*.n_nodes      =*/0,
      /*.n_leafs      =*/0,
      /*.nodes        =*/{NULL},
      /*.grads        =*/{NULL},
      /*.leafs        =*/{NULL},
      /*.hash_table   =*/{NULL},
      /*.order        =*/GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT,
      /*.perf_runs    =*/0,
      /*.perf_cycles  =*/0,
      /*.perf_time_us =*/0,
  };

  return cgraph;
}

static struct ggml_cgraph* llama_build_graph(llama_context* lctx, const llama_batch& batch) {
  const auto* model = lctx->model;
  const auto& hparams = model->hparams;
  const auto& cparams = lctx->cparams;

  GGML_ASSERT(!!lctx->kv_self.ctx);

  const int64_t n_embd = hparams.n_embd;
  const int64_t n_layer = hparams.n_layer;
  const int64_t n_ctx = cparams.n_ctx;
  const int64_t n_head = hparams.n_head;
  const int64_t n_head_kv = hparams.n_head_kv;
  const int64_t n_embd_head = hparams.n_embd_head();
  const int64_t n_embd_gqa = hparams.n_embd_gqa();

  GGML_ASSERT(n_embd_head == hparams.n_rot);

  const float freq_base = cparams.rope_freq_base;
  const float freq_scale = cparams.rope_freq_scale;
  const float norm_rms_eps = hparams.f_norm_rms_eps;

  const int n_gpu_layers = model->n_gpu_layers;

  const int64_t n_tokens = batch.n_tokens;
  const int32_t n_kv = lctx->allocator->measure ? n_ctx : lctx->kv_self.n;
  const int32_t kv_head = lctx->allocator->measure ? n_ctx - n_tokens : lctx->kv_self.head;

  const bool do_rope_shift = lctx->allocator->measure || lctx->kv_self.has_shift;

  // printf("n_kv = %d\n", n_kv);

  struct ggml_init_params params = {
      /*.mem_size   =*/lctx->buf_compute.size,
      /*.mem_buffer =*/lctx->buf_compute.data,
      /*.no_alloc   =*/false,
  };

  params.no_alloc = true;

  struct ggml_context* ctx0 = ggml_init(params);

  ggml_cgraph* gf = ggml_new_graph(ctx0);

  struct ggml_tensor* cur;
  struct ggml_tensor* inpL;

  if (batch.token) {
    struct ggml_tensor* inp_tokens = ggml_new_tensor(ctx0, GGML_TYPE_I32, 1, &n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");

    ggml_allocator_alloc(lctx->allocator, inp_tokens);
    if (!lctx->allocator->measure) {
      memcpy(inp_tokens->data, batch.token, n_tokens * type_traits[inp_tokens->type].type_size);
    }

    inpL = ggml_get_rows(ctx0, model->tok_embeddings, inp_tokens);
  } else {
    inpL = ggml_new_tensor(ctx0, GGML_TYPE_F32, 2, new int64_t[2]{n_embd, n_tokens});

    ggml_allocator_alloc(lctx->allocator, inpL);
    if (!lctx->allocator->measure) {
      memcpy(inpL->data, batch.embd, n_tokens * n_embd * type_traits[inpL->type].type_size);
    }
  }

  const int i_gpu_start = n_layer - n_gpu_layers;
  (void)i_gpu_start;

  // offload functions set the tensor output backend to GPU
  // tensors are GPU-accelerated if any input or the output has been offloaded
  offload_func_t offload_func_nr = llama_nop;  // nr = non-repeating
  offload_func_t offload_func_kq = llama_nop;
  offload_func_t offload_func_v = llama_nop;

  if (ggml_use_cublas()) {
    if (n_gpu_layers > n_layer) {
      offload_func_nr = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 1) {
      offload_func_v = ggml_cuda_assign_buffers_no_alloc;
    }
    if (n_gpu_layers > n_layer + 2) {
      offload_func_kq = ggml_cuda_assign_buffers_no_alloc;
    }
  }

  // KQ_scale
  struct ggml_tensor* KQ_scale = ggml_new_tensor(ctx0, GGML_TYPE_F32, 1, new int64_t[1]{1});
  ggml_set_name(KQ_scale, "1/sqrt(n_embd_head)");
  ggml_allocator_alloc(lctx->allocator, KQ_scale);
  if (!lctx->allocator->measure) {
    ggml_set_f32(KQ_scale, 1.0f / sqrtf(float(n_embd_head)));
  }

  // KQ_mask (mask for 1 head, it will be broadcasted to all heads)
  struct ggml_tensor* KQ_mask = ggml_new_tensor(ctx0, GGML_TYPE_F32, 3, new int64_t[3]{n_kv, n_tokens, 1});
  offload_func_kq(KQ_mask);
  ggml_set_name(KQ_mask, "KQ_mask");
  ggml_allocator_alloc(lctx->allocator, KQ_mask);
  if (!lctx->allocator->measure) {
    float* data = (float*)KQ_mask->data;
    memset(data, 0, ggml_nbytes(KQ_mask));

    for (int h = 0; h < 1; ++h) {
      for (int j = 0; j < n_tokens; ++j) {
        const llama_pos pos = batch.pos[j];
        const llama_seq_id seq_id = batch.seq_id[j];

        for (int i = 0; i < n_kv; ++i) {
          if (!lctx->kv_self.cells[i].has_seq_id(seq_id) || lctx->kv_self.cells[i].pos > pos) {
            data[h * (n_kv * n_tokens) + j * n_kv + i] = -INFINITY;
          }
        }
      }
    }
  }

  // KQ_pos - contains the positions
  struct ggml_tensor* KQ_pos = ggml_new_tensor(ctx0, GGML_TYPE_I32, 1, new int64_t[1]{n_tokens});
  offload_func_kq(KQ_pos);
  ggml_set_name(KQ_pos, "KQ_pos");
  ggml_allocator_alloc(lctx->allocator, KQ_pos);
  if (!lctx->allocator->measure) {
    int* data = (int*)KQ_pos->data;
    for (int i = 0; i < n_tokens; ++i) {
      data[i] = batch.pos[i];
    }
  }

  // shift the entire K-cache if needed
  if (do_rope_shift) {
    struct ggml_tensor* K_shift = ggml_new_tensor(ctx0, GGML_TYPE_I32, 1, new int64_t[1]{n_ctx});
    offload_func_kq(K_shift);
    ggml_set_name(K_shift, "K_shift");
    ggml_allocator_alloc(lctx->allocator, K_shift);
    if (!lctx->allocator->measure) {
      int* data = (int*)K_shift->data;
      for (int i = 0; i < n_ctx; ++i) {
        data[i] = lctx->kv_self.cells[i].delta;
      }
    }

    for (int il = 0; il < n_layer; ++il) {
      struct ggml_tensor* tmp_tensor =
          ggml_rope_custom_inplace(ctx0,
                                   ggml_view_3d(ctx0, lctx->kv_self.k, n_embd_head, n_head_kv, n_ctx,
                                                type_traits[lctx->kv_self.k->type].type_size * n_embd_head,
                                                type_traits[lctx->kv_self.k->type].type_size * n_embd_gqa,
                                                type_traits[lctx->kv_self.k->type].type_size * n_embd_gqa * n_ctx * il),
                                   K_shift, n_embd_head, 0, 0, freq_base, freq_scale);
      offload_func_kq(tmp_tensor);
      ggml_build_forward_impl(gf, tmp_tensor, true);
    }
  }

  for (int il = 0; il < n_layer; ++il) {
    ggml_format_name(inpL, "layer_inp_%d", il);

    offload_func_t offload_func = llama_nop;

    if (ggml_use_cublas() && (il >= i_gpu_start)) {
      offload_func = ggml_cuda_assign_buffers_no_alloc;
    }

    struct ggml_tensor* inpSA = inpL;

    // norm
    {
      cur = ggml_rms_norm(ctx0, inpL, norm_rms_eps);
      offload_func(cur);
      ggml_set_name(cur, "rms_norm_0");

      // cur = cur*attn_norm(broadcasted)
      cur = ggml_mul(ctx0, cur, model->layers[il].attn_norm);
      offload_func(cur);
      ggml_set_name(cur, "attention_norm_0");
    }

    // self-attention
    {
      // compute Q and K and RoPE them
      struct ggml_tensor* tmpk = ggml_mul_mat(ctx0, model->layers[il].wk, cur);
      offload_func_kq(tmpk);
      ggml_set_name(tmpk, "tmpk");

      struct ggml_tensor* tmpq = ggml_mul_mat(ctx0, model->layers[il].wq, cur);
      offload_func_kq(tmpq);
      ggml_set_name(tmpq, "tmpq");

      struct ggml_tensor* Kcur = ggml_rope_custom(ctx0, ggml_reshape_3d(ctx0, tmpk, n_embd_head, n_head_kv, n_tokens),
                                                  KQ_pos, n_embd_head, 0, 0, freq_base, freq_scale);
      offload_func_kq(Kcur);
      ggml_set_name(Kcur, "Kcur");

      struct ggml_tensor* Qcur = ggml_rope_custom(ctx0, ggml_reshape_3d(ctx0, tmpq, n_embd_head, n_head, n_tokens),
                                                  KQ_pos, n_embd_head, 0, 0, freq_base, freq_scale);
      offload_func_kq(Qcur);
      ggml_set_name(Qcur, "Qcur");

      // store key and value to memory
      {
        // compute the transposed [n_tokens, n_embd] V matrix

        struct ggml_tensor* tmpv = ggml_mul_mat(ctx0, model->layers[il].wv, cur);
        offload_func_v(tmpv);
        ggml_set_name(tmpv, "tmpv");

        struct ggml_tensor* Vcur = ggml_transpose(ctx0, ggml_reshape_2d(ctx0, tmpv, n_embd_gqa, n_tokens));
        offload_func_v(Vcur);
        ggml_set_name(Vcur, "Vcur");

        struct ggml_tensor* k =
            ggml_view_1d(ctx0, lctx->kv_self.k, n_tokens * n_embd_gqa,
                         (type_traits[lctx->kv_self.k->type].type_size * n_embd_gqa) * (il * n_ctx + kv_head));
        offload_func_kq(k);
        ggml_set_name(k, "k");

        struct ggml_tensor* v = ggml_view_2d(ctx0, lctx->kv_self.v, n_tokens, n_embd_gqa,
                                             (n_ctx)*type_traits[lctx->kv_self.v->type].type_size,
                                             (il * n_ctx) * type_traits[lctx->kv_self.v->type].type_size * n_embd_gqa +
                                                 kv_head * type_traits[lctx->kv_self.v->type].type_size);
        offload_func_v(v);
        ggml_set_name(v, "v");

        // important: storing RoPE-ed version of K in the KV cache!
        ggml_build_forward_impl(gf, ggml_cpy(ctx0, Kcur, k), true);
        ggml_build_forward_impl(gf, ggml_cpy(ctx0, Vcur, v), true);
      }

      struct ggml_tensor* Q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);
      offload_func_kq(Q);
      ggml_set_name(Q, "Q");

      struct ggml_tensor* K = ggml_view_3d(ctx0, lctx->kv_self.k, n_embd_head, n_kv, n_head_kv,
                                           type_traits[lctx->kv_self.k->type].type_size * n_embd_gqa,
                                           type_traits[lctx->kv_self.k->type].type_size * n_embd_head,
                                           type_traits[lctx->kv_self.k->type].type_size * n_embd_gqa * n_ctx * il);
      offload_func_kq(K);
      ggml_set_name(K, "K");

      // K * Q
      struct ggml_tensor* KQ = ggml_mul_mat(ctx0, K, Q);
      offload_func_kq(KQ);
      ggml_set_name(KQ, "KQ");

      // KQ_scaled = KQ / sqrt(n_embd_head)
      // KQ_scaled shape [n_kv, n_tokens, n_head, 1]
      struct ggml_tensor* KQ_scaled = ggml_scale(ctx0, KQ, KQ_scale);
      offload_func_kq(KQ_scaled);
      ggml_set_name(KQ_scaled, "KQ_scaled");

      // KQ_masked = mask_past(KQ_scaled)
      struct ggml_tensor* KQ_masked = ggml_add(ctx0, KQ_scaled, KQ_mask);
      offload_func_kq(KQ_masked);
      ggml_set_name(KQ_masked, "KQ_masked");

      // KQ = soft_max(KQ_masked)
      struct ggml_tensor* KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
      offload_func_v(KQ_soft_max);
      ggml_set_name(KQ_soft_max, "KQ_soft_max");

      // split cached V into n_head heads
      struct ggml_tensor* V = ggml_view_3d(ctx0, lctx->kv_self.v, n_kv, n_embd_head, n_head_kv,
                                           type_traits[lctx->kv_self.v->type].type_size * n_ctx,
                                           type_traits[lctx->kv_self.v->type].type_size * n_ctx * n_embd_head,
                                           type_traits[lctx->kv_self.v->type].type_size * n_ctx * n_embd_gqa * il);
      offload_func_v(V);
      ggml_set_name(V, "V");

#if 1
      struct ggml_tensor* KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);
      offload_func_v(KQV);
      ggml_set_name(KQV, "KQV");
#else
      // make V contiguous in memory to speed up the matmul, however we waste time on the copy
      // on M1 this is faster for the perplexity computation, but ~5% slower for the single-token generation
      // is there a better way?
      struct ggml_tensor* V_cont =
          ggml_cpy(ctx0, V, ggml_new_tensor_3d(ctx0, lctx->kv_self.v->type, n_ctx, n_embd_head, n_head));
      struct ggml_tensor* KQV = ggml_mul_mat(ctx0, V_cont, KQ_soft_max);
#endif

      // KQV_merged = KQV.permute(0, 2, 1, 3)
      struct ggml_tensor* KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
      offload_func_v(KQV_merged);
      ggml_set_name(KQV_merged, "KQV_merged");

      // cur = KQV_merged.contiguous().view(n_embd, n_tokens)
      cur = ggml_cont_2d(ctx0, KQV_merged, n_embd, n_tokens);
      offload_func_v(cur);
      ggml_set_name(cur, "KQV_merged_contiguous");

      // projection (no bias)
      cur = ggml_mul_mat(ctx0, model->layers[il].wo, cur);
      offload_func(cur);
      ggml_set_name(cur, "result_wo");
    }

    struct ggml_tensor* inpFF = ggml_add(ctx0, cur, inpSA);
    offload_func(inpFF);
    ggml_set_name(inpFF, "inpFF");

    // feed-forward network
    {
      // norm
      {
        cur = ggml_rms_norm(ctx0, inpFF, norm_rms_eps);
        offload_func(cur);
        ggml_set_name(cur, "rms_norm_1");

        // cur = cur*ffn_norm(broadcasted)
        cur = ggml_mul(ctx0, cur, model->layers[il].ffn_norm);
        offload_func(cur);
        ggml_set_name(cur, "ffn_norm");
      }

      struct ggml_tensor* tmp = ggml_mul_mat(ctx0, model->layers[il].w3, cur);
      offload_func(tmp);
      ggml_set_name(tmp, "result_w3");

      cur = ggml_mul_mat(ctx0, model->layers[il].w1, cur);
      offload_func(cur);
      ggml_set_name(cur, "result_w1");

      // SILU activation
      cur = ggml_silu(ctx0, cur);
      offload_func(cur);
      ggml_set_name(cur, "silu");

      cur = ggml_mul(ctx0, cur, tmp);
      offload_func(cur);
      ggml_set_name(cur, "silu_x_result_w3");

      cur = ggml_mul_mat(ctx0, model->layers[il].w2, cur);
      offload_func(cur);
      ggml_set_name(cur, "result_w2");
    }

    cur = ggml_add(ctx0, cur, inpFF);
    offload_func(cur);
    ggml_set_name(cur, "inpFF_+_result_w2");

    // input for next layer
    inpL = cur;
  }

  cur = inpL;

  // norm
  {
    cur = ggml_rms_norm(ctx0, cur, norm_rms_eps);
    offload_func_nr(cur);
    ggml_set_name(cur, "rms_norm_2");

    // cur = cur*norm(broadcasted)
    cur = ggml_mul(ctx0, cur, model->output_norm);
    // offload_func_nr(cur); // TODO CPU + GPU mirrored backend
    ggml_set_name(cur, "result_norm");
  }

  // lm_head
  cur = ggml_mul_mat(ctx0, model->output, cur);
  ggml_set_name(cur, "result_output");

  ggml_build_forward_impl(gf, cur, true);

  ggml_free(ctx0);

  return gf;
}

struct llama_context* llama_new_context(llama_model* model, struct llama_context_params params) {
  if (!model) {
    return nullptr;
  }

  llama_context* lctx = new llama_context(model);

  const auto& hparams = model->hparams;
  auto& cparams = lctx->cparams;

  cparams.n_batch = params.n_batch;
  cparams.n_ctx = params.n_ctx == 0 ? hparams.n_ctx_train : params.n_ctx;
  cparams.rope_freq_base = params.rope_freq_base == 0 ? hparams.rope_freq_base_train : params.rope_freq_base;
  cparams.rope_freq_scale = params.rope_freq_scale == 0 ? hparams.rope_freq_scale_train : params.rope_freq_scale;
  cparams.n_threads = params.n_threads;
  cparams.n_threads_batch = params.n_threads_batch;
  cparams.mul_mat_q = params.mul_mat_q;

  if (params.seed == LLAMA_DEFAULT_SEED) {
    params.seed = time(NULL);
  }

  LLAMA_LOG_INFO("%s: n_ctx      = %u\n", __func__, cparams.n_ctx);
  LLAMA_LOG_INFO("%s: freq_base  = %.1f\n", __func__, cparams.rope_freq_base);
  LLAMA_LOG_INFO("%s: freq_scale = %g\n", __func__, cparams.rope_freq_scale);

  lctx->rng = std::mt19937(params.seed);
  lctx->logits_all = params.logits_all;

  ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

  // reserve memory for context buffers
  if (!hparams.vocab_only) {
    if (!llama_kv_cache_init(model->hparams, lctx->kv_self, memory_type, cparams.n_ctx, model->n_gpu_layers)) {
      LLAMA_LOG_ERROR("%s: llama_kv_cache_init() failed for self-attention cache\n", __func__);
      delete lctx;
      return nullptr;
    }

    {
      const size_t memory_size = ggml_nbytes(lctx->kv_self.k) + ggml_nbytes(lctx->kv_self.v);
      LLAMA_LOG_INFO("%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
    }

    // resized during inference
    if (params.logits_all) {
      lctx->logits.reserve(cparams.n_ctx * hparams.n_vocab);
    } else {
      lctx->logits.reserve(hparams.n_vocab);
    }

    if (params.embedding) {
      lctx->embedding.resize(hparams.n_embd);
    }

    {
      static const size_t tensor_alignment = 32;
      // the compute buffer is used to store the tensor and graph structs, while the allocator buffer is used for the
      // tensor data
      lctx->buf_compute.resize(ggml_tensor_overhead() * GGML_MAX_NODES + ggml_graph_overhead());

      // create measure allocator
      lctx->allocator = ggml_allocator_new_measure(tensor_alignment);

      // build worst-case graph
      int n_tokens = (int)std::min(cparams.n_ctx, cparams.n_batch);
      int n_past = cparams.n_ctx - n_tokens;
      llama_token token = model->vocab.special_bos_id;  // not actually used by llama_build_graph, but required to
                                                        // choose between token and embedding inputs graph

      struct llama_batch lbatch = {
          /*n_tokens    =*/n_tokens,
          /*tokens      =*/&token,
          /*embd        =*/nullptr,
          /*pos         =*/nullptr,
          /*seq_id      =*/nullptr,
          /*logits      =*/nullptr,
          /*all_pos_0   =*/n_past,
          /*all_pos_1   =*/1,
          /*all_seq_id  =*/0,
      };
      ggml_cgraph* gf = llama_build_graph(lctx, lbatch);

      // measure memory requirements for the graph
      size_t alloc_size = ggml_allocator_alloc_graph(lctx->allocator, gf) + tensor_alignment;

      LLAMA_LOG_INFO("%s: compute buffer total size = %.2f MB\n", __func__,
                     (lctx->buf_compute.size + alloc_size) / 1024.0 / 1024.0);

      // recreate allocator with exact memory requirements
      ggml_allocator_free(lctx->allocator);
      lctx->buf_alloc.resize(alloc_size);
      lctx->allocator = ggml_allocator_new(lctx->buf_alloc.data, lctx->buf_alloc.size, tensor_alignment);

      if (ggml_use_cublas()) {
        ggml_cuda_set_scratch_size(alloc_size);
        LLAMA_LOG_INFO("%s: VRAM scratch buffer: %.2f MB\n", __func__, alloc_size / 1024.0 / 1024.0);

        // calculate total VRAM usage
        auto add_tensor = [](const ggml_tensor* t, size_t& size) {
          if (t->backend == GGML_BACKEND_GPU || t->backend == GGML_BACKEND_GPU_SPLIT) {
            size += ggml_nbytes(t);
          }
        };
        size_t model_vram_size = 0;
        for (const auto& kv : model->tensors_by_name) {
          add_tensor(kv.second, model_vram_size);
        }

        size_t kv_vram_size = 0;
        add_tensor(lctx->kv_self.k, kv_vram_size);
        add_tensor(lctx->kv_self.v, kv_vram_size);

        size_t ctx_vram_size = alloc_size + kv_vram_size;
        size_t total_vram_size = model_vram_size + ctx_vram_size;

        LLAMA_LOG_INFO("%s: total VRAM used: %.2f MB (model: %.2f MB, context: %.2f MB)\n", __func__,
                       total_vram_size / 1024.0 / 1024.0, model_vram_size / 1024.0 / 1024.0,
                       ctx_vram_size / 1024.0 / 1024.0);
      }
    }
  }

  return lctx;
}

// decode a batch of tokens by evaluating the transformer
//
//   - lctx:      llama context
//   - batch:     batch to evaluate
//   - n_threads: number of threads to use
//
// return 0 on success
// return positive int on warning
// return negative int on error
//
int llama_decode(llama_context* lctx, llama_batch& batch) {
  const uint32_t n_tokens = batch.n_tokens;

  if (n_tokens == 0) {
    LLAMA_LOG_ERROR("%s: n_tokens == 0", __func__);
    return -1;
  }

  const auto* model = lctx->model;
  const auto& hparams = model->hparams;
  const auto& cparams = lctx->cparams;

  const auto n_batch = cparams.n_batch;

  GGML_ASSERT(n_tokens <= n_batch);

  int n_threads = n_tokens == 1 ? cparams.n_threads : cparams.n_threads_batch;
  GGML_ASSERT((!batch.token && batch.embd) || (batch.token && !batch.embd));  // NOLINT

  const int64_t t_start_us = ggml_time_us();

  GGML_ASSERT(n_threads > 0);
  GGML_ASSERT(!!lctx->kv_self.ctx);

  const int64_t n_embd = hparams.n_embd;
  const int64_t n_vocab = hparams.n_vocab;

  // helpers for smoother batch API transistion
  // after deprecating the llama_eval calls, these will be removed
  std::vector<llama_pos> pos;
  std::vector<llama_seq_id> seq_id;

  if (batch.pos == nullptr) {
    pos.resize(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
      pos[i] = batch.all_pos_0 + i * batch.all_pos_1;
    }

    batch.pos = pos.data();
  }

  if (batch.seq_id == nullptr) {
    seq_id.resize(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
      seq_id[i] = batch.all_seq_id;
    }

    batch.seq_id = seq_id.data();
  }

  // we always start to search for a free slot from the start of the cache
  // TODO: better strategies can be implemented
  lctx->kv_self.head = 0;

  if (!llama_kv_cache_find_slot(lctx->kv_self, batch)) {
    return 1;
  }

  // a heuristic, to avoid attending the full cache if it is not yet utilized
  // after enough generations, the benefit from this heuristic disappears
  // if we start defragmenting the cache, the benefit from this will be more important
  // lctx->kv_self.n = std::max(32, GGML_PAD(llama_kv_cache_cell_max(lctx->kv_self), 32));   // TODO: this might be
  // better for CUDA?
  lctx->kv_self.n = std::min((int32_t)cparams.n_ctx, std::max(32, llama_kv_cache_cell_max(lctx->kv_self)));

  // printf("lctx->kv_self.n = %d\n", lctx->kv_self.n);

  ggml_allocator_reset(lctx->allocator);

  ggml_cgraph* gf = llama_build_graph(lctx, batch);

  // Examine compute graph
  printf("\nExamine compute graph\n");
  for (size_t i = 0; i < gf->n_nodes; i++) {
    const auto node = gf->nodes[i];
    printf("Node: %-40s op: %s\n", node->name, GGML_OP_NAMES[node->op]);
  }
  for (size_t i = 0; i < gf->n_leafs; i++) {
    const auto leaf = gf->leafs[i];
    printf("Leaf: %-40s op: %s\n", leaf->name, GGML_OP_NAMES[leaf->op]);
  }

  ggml_allocator_alloc_graph(lctx->allocator, gf);

  if (ggml_use_cublas()) {
    for (int i = 0; i < gf->n_leafs; i++) {
      ggml_tensor* node = gf->leafs[i];
      if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
        ggml_cuda_assign_scratch_offset(node, (char*)node->data - (char*)lctx->buf_alloc.data);
        ggml_cuda_copy_to_device(node);
      }
    }

    for (int i = 0; i < gf->n_nodes; i++) {
      ggml_tensor* node = gf->nodes[i];
      if (node->backend == GGML_BACKEND_GPU && node->extra == NULL) {
        ggml_cuda_assign_scratch_offset(node, (char*)node->data - (char*)lctx->buf_alloc.data);
      }
    }

    ggml_cuda_set_mul_mat_q(cparams.mul_mat_q);
  }

  // LLAMA_LOG_INFO("graph build time: %.3f ms (%d nodes, %d leafs)\n", (ggml_time_us() - t_start_us)/1000.0,
  // gf->n_nodes, gf->n_leafs);

  // for big prompts, if BLAS is enabled, it is better to use only one thread
  // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
  // TODO: this is mostly important for Apple Silicon where CBLAS is still performing very well
  //       we still need some threads to process all non-mul_mat ops, but not too much to avoid interfering
  //       with the BLAS calls. need a better solution
  if (n_tokens >= 32 && ggml_use_cublas()) {
    n_threads = std::min(4, n_threads);
  }

  // If all tensors can be run on the GPU then using more than 1 thread is detrimental.
  const bool full_offload_supported =
      model->arch == LLM_ARCH_LLAMA || model->arch == LLM_ARCH_BAICHUAN || model->arch == LLM_ARCH_FALCON;
  const bool fully_offloaded = model->n_gpu_layers >= (int)hparams.n_layer + 3;
  if (ggml_use_cublas() && full_offload_supported && fully_offloaded) {
    n_threads = 1;
  }

  struct ggml_tensor* res = gf->nodes[gf->n_nodes - 1];
  struct ggml_tensor* embeddings = gf->nodes[gf->n_nodes - 2];

  GGML_ASSERT(strcmp(res->name, "result_output") == 0);
  GGML_ASSERT(strcmp(embeddings->name, "result_norm") == 0);

  // ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);
  {
    struct ggml_cplan plan = ggml_graph_plan(gf, n_threads);

    if (plan.work_size > 0) {
      lctx->work_buffer.resize(plan.work_size);
      plan.work_data = lctx->work_buffer.data();
    }

    ggml_graph_compute(gf, &plan);
  }

  // update the kv ring buffer
  lctx->kv_self.head += n_tokens;
  lctx->kv_self.has_shift = false;

  if (ggml_show_perf()) {
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_PERF to be defined
    ggml_graph_print(gf);
  }

  // plot the computation graph in dot format (for debugging purposes)
  // if (n_past%100 == 0) {
  //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
  //}

  // extract logits
  {
    auto& logits_out = lctx->logits;

    if (batch.logits) {
      logits_out.resize(n_vocab * n_tokens);
      for (uint32_t i = 0; i < n_tokens; i++) {
        if (batch.logits[i] == 0) {
          continue;
        }
        memcpy(logits_out.data() + (n_vocab * i), (float*)res->data + (n_vocab * i), sizeof(float) * n_vocab);
      }
    } else if (lctx->logits_all) {
      logits_out.resize(n_vocab * n_tokens);
      memcpy(logits_out.data(), (float*)res->data, sizeof(float) * n_vocab * n_tokens);
    } else {
      logits_out.resize(n_vocab);
      memcpy(logits_out.data(), (float*)res->data + (n_vocab * (n_tokens - 1)), sizeof(float) * n_vocab);
    }
  }

  // extract embeddings
  if (!lctx->embedding.empty()) {
    auto& embedding_out = lctx->embedding;

    embedding_out.resize(n_embd);
    memcpy(embedding_out.data(), (float*)embeddings->data + (n_embd * (n_tokens - 1)), sizeof(float) * n_embd);
  }

  // measure the performance only for the single-token evals
  if (n_tokens == 1) {
    lctx->t_eval_us += ggml_time_us() - t_start_us;
    lctx->n_eval++;
  } else if (n_tokens > 1) {
    lctx->t_p_eval_us += ggml_time_us() - t_start_us;
    lctx->n_p_eval += n_tokens;
  }

  // get a more accurate load time, upon first eval
  // TODO: fix this
  if (!lctx->has_evaluated_once) {
    lctx->t_load_us = ggml_time_us() - lctx->t_start_us;
    lctx->has_evaluated_once = true;
  }

  return 0;
}

// =======================================================================
// Global variables
// =======================================================================
llama_model* model = new llama_model();
llama_ftype ftype;
std::string model_arch_name;
int64_t n_elements = 0;
size_t n_bytes = 0;
size_t n_tensors = 0;
size_t n_created = 0;

#define __MAIN_FUNCTION__
// =======================================================================
// Main
// =======================================================================
int main() {
  init_type_traits();

  struct ggml_context* ctx_meta = NULL;

  // Hard-coded params
  const char* model_file_path = "models/llama-2-13b-chat.Q4_0.gguf";
  const bool use_mmap = true;
  const bool use_mlock = false;
  const int n_gpu_layers = 0;
  const size_t n_batch = 512;
  const size_t n_threads = 1;
  const size_t n_threads_batch = 1;
  float* tensor_split = nullptr;

  printf("%s: llama backend init\n", __func__);
  bool numa = false;
  llama_backend_init(numa);

  int main_gpu = 0;
  if (ggml_use_cublas()) {
    LLAMA_LOG_INFO("%s: using CUDA for GPU acceleration\n", __func__);
    ggml_cuda_set_main_device(main_gpu);
  }

#define P_BUILD_GGUF_CONTEXT
  FILE* file = nullptr;
  uint32_t magic = 0;
  size_t offset = 0;
  size_t nitems = 0;
  size_t ret = 0;
  struct gguf_context* gguf_ctx = (struct gguf_context*)ggml_aligned_malloc(sizeof(struct gguf_context));

#define __OPEN_MODEL_FILE
  // Open model file
  {
    printf("== Open model file ==\n");
    file = fopen(model_file_path, "rb");
    if (!file) {
      throw std::runtime_error(format("%s:%d: Could not read %s\n", __FILE__, __LINE__, model_file_path));
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
    printf("Data offset = %ld\n", gguf_ctx->offset);
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
      if (kid != -1) {
        hparams.n_head_kv = gguf_ctx->kv[kid].value.uint32;
      }
      LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_head_kv:", hparams.n_head_kv);

      // rope_freq_base (optional)
      hparams.rope_freq_base_train = 10000.0f;
      key = build_llm_key(LLM_KV_NAMES[LLM_KV_ROPE_FREQ_BASE], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid != -1) {
        hparams.rope_freq_base_train = gguf_ctx->kv[kid].value.float32;
      }
      LLAMA_LOG_INFO("%s: - %42s %-20.3lf\n", __func__,
                     "model->hparams.rope_freq_base_train:", hparams.rope_freq_base_train);

      // rope_freq_scale (inverse of the kv) is optional
      float ropescale = 1.0f;
      key = build_llm_key(LLM_KV_NAMES[LLM_KV_ROPE_SCALE_LINEAR], model_arch_name);
      kid = gguf_find_key(gguf_ctx, key.c_str());
      if (kid != -1) {
        ropescale = gguf_ctx->kv[kid].value.float32;
      }
      hparams.rope_freq_scale_train = 1.0f / ropescale;
      LLAMA_LOG_INFO("%s: - %42s %-20.3lf\n", __func__,
                     "model->hparams.rope_freq_scale_train:", hparams.rope_freq_scale_train);

      // Sanity check for n_rot
      {
        hparams.n_rot = hparams.n_embd / hparams.n_head;
        key = build_llm_key(LLM_KV_NAMES[LLM_KV_ROPE_DIMENSION_COUNT], model_arch_name);
        kid = gguf_find_key(gguf_ctx, key.c_str());
        if (kid != -1) {
          hparams.n_rot = gguf_ctx->kv[kid].value.uint32;
        }

        if (model->arch == LLM_ARCH_LLAMA) {
          if (hparams.n_rot != hparams.n_embd / hparams.n_head) {
            throw std::runtime_error(
                format("invalid n_rot: %u, expected %u", hparams.n_rot, hparams.n_embd / hparams.n_head));
          }
        }

        LLAMA_LOG_INFO("%s: - %42s %-20ld\n", __func__, "model->hparams.n_rot:", hparams.n_rot);
      }

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
      if (ggml_use_cublas()) {
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

    if (ggml_use_cublas()) {
      ggml_cuda_set_tensor_split(tensor_split);
    }

    printf("\n");

#define __DATA_FOR_TENSORS
    printf("Loading data for tensors\n");
    std::unique_ptr<llama_mmap> mapping;
    llama_progress_callback progress_callback = NULL;
    void* progress_callback_user_data = NULL;
    {
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

      // mmap model file
      llama_mlock* lmlock = use_mlock ? &model->mlock_mmap : NULL;
      llama_file file(model_file_path, "rb");
      if (use_mmap) {
        mapping.reset(new llama_mmap(&file, size_pref, ggml_is_numa()));
        if (lmlock) {
          lmlock->init(mapping->addr);
        }
      }

      // Set up progress callback
      unsigned cur_percentage = 0;
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

      size_t done_size = 0;
      for (int i = 0; i < n_tensors; i++) {
        struct ggml_tensor* cur = ggml_get_tensor(model->ctx, gguf_ctx->infos[i].name.data);
        GGML_ASSERT(cur);  // unused tensors should have been caught by load_data already

        // Load data for cur
        const size_t data_offset = gguf_ctx->offset;
        const size_t tensor_offset = gguf_ctx->infos[i].offset;
        cur->data = (uint8_t*)mapping->addr + (data_offset + tensor_offset);
        std::string name = cur->name;
        printf("%-40s: %p = %p + %ld\n", name.c_str(), cur->data, mapping->addr, (data_offset + tensor_offset));

        switch (cur->backend) {
          case GGML_BACKEND_CPU:
            if (use_mmap && lmlock) {
              size_lock += ggml_nbytes(cur);
              lmlock->grow_to(size_lock);
            }
            break;
          case GGML_BACKEND_GPU:
          case GGML_BACKEND_GPU_SPLIT:
            // TODO: test if this works !!
            ggml_cuda_transform_tensor(cur->data, cur);
            if (!use_mmap) {
              free(cur->data);
            }
            break;
          default:
            continue;
        }

        done_size += ggml_nbytes(cur);
        progress_callback((float)done_size / size_data, progress_callback_user_data);
      }
    }

    if (progress_callback) {
      progress_callback(1.0f, progress_callback_user_data);
    }

    model->mapping = std::move(mapping);

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model->t_load_us = ggml_time_us() - model->t_start_us;
    printf("\nOK\n\n");
  }

#define __CLEAN_UP_MEMORY
  {
    if (gguf_ctx) {
      gguf_free(gguf_ctx);
    }
    if (ctx_meta) {
      ggml_free(ctx_meta);
    }
  }

#define P_WARM_UP_MODEL
  {
    LLAMA_LOG_INFO("Warming up the model with an empty run\n");
    llama_context_params lparams = llama_context_default_params();
    lparams.n_threads = 1;
    lparams.n_threads_batch = 1;
    llama_context* lctx = llama_new_context(model, lparams);

    std::vector<llama_token> tmp = {
        model->vocab.special_bos_id,
        model->vocab.special_eos_id,
    };
    struct llama_batch lbatch = {
        /*n_tokens    =*/(int32_t)std::min(tmp.size(), (size_t)n_batch),
        /*tokens      =*/tmp.data(),
        /*embd        =*/nullptr,
        /*pos         =*/nullptr,
        /*seq_id      =*/nullptr,
        /*logits      =*/nullptr,
        /*all_pos_0   =*/0,
        /*all_pos_1   =*/1,
        /*all_seq_id  =*/0,
    };

    llama_decode(lctx, lbatch);
    llama_kv_cache_tokens_rm(lctx->kv_self, -1, -1);
    llama_reset_timings(lctx);
  }
}