# =====================================
# Motivation
# =====================================
I decided to understand this software comprehensively.  
Read code linearly part by part. Determine!  
Go to the edge of it.

# =====================================
# Machine Learning
# =====================================
## In machine learning, what does logits mean?
In machine learning, the term "logits" refers to the raw, unnormalized output scores produced by a classification model before they are transformed into probabilities through a softmax function. Logits are typically used in the context of multi-class classification problems.

Here's how the process generally works:

1. The model makes predictions for each class or category in a classification problem.  
2. These predictions are represented as logits, which are typically real numbers and can be positive, negative, or zero.  
3. To convert logits into probabilities, a softmax function is applied to the logits. The softmax function exponentiates the logits and then normalizes them, ensuring that the resulting values sum to 1. This transformation makes it easier to interpret the model's output as probabilities.  
The softmax function is defined as follows for class j:

P(class j) = exp(logits_j) / sum(exp(logits_i) for all classes i)

The class with the highest probability after the softmax transformation is typically considered the model's predicted class.

In summary, logits are the intermediate values in a classification model that represent the unnormalized scores for each class before they are turned into probabilities for making a final prediction.

# =====================================
# CUDA
# =====================================
## NVCC Options for Separate Compilation
CUDA works by embedding device code into host objects. In whole program compilation, it embeds executable device code into the host object. In separate compilation, we embed relocatable device code into the host object, and run nvlink, the device linker, to link all the device code together. The output of nvlink is then linked together with all the host objects by the host linker to form the final executable.

The generation of relocatable vs executable device code is controlled by the --relocatable-device-code option.

The --compile option is already used to control stopping a compile at a host object, so a new option --device-c is added that simply does --relocatable-device-code=true --compile.

To invoke just the device linker, the --device-link option can be used, which emits a host object containing the embedded executable device code. The output of that must then be passed to the host linker. Or:

nvcc <objects>
can be used to implicitly call both the device and host linkers. This works because if the device linker does not see any relocatable code it does not do anything.

The following figure shows the flow.

![CUDA Separate Compilation Trajectory](./images/nvcc-options-for-separate-compilation.png)

CUDA Separate Compilation Trajectory

## CUDA C++ Programming
Ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

A routine decorated with host instructs the compiler to generate a host-callable entry point (i.e. compile it as host code). Such a routine is host code that can only be called from other host code.

A routine decorated with device instructs the compiler to generate a device-callable entry point (i.e. compile it as device code). Such a routine is device code that can only be called from other device code.

You can use both. If you use both, order does not matter. If you use both, the compiler generates both types of routines describe as above, one with a device-callable entry point, and one with a host-callable entry point.

The only situation where device code can be “called from host code” is the kernel launch, which must be decorated with global

[url]https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers[/url] 147

A routine with none of the above decorations is treated by nvcc implicitly as if it were decorated with host

### CUDA C++: Launch
```bash
sudo apt-get install libncurses5
```

# =====================================
# llama.cpp
# =====================================
## Build
```bash
export LLAMA_DEBUG=1
export LLAMA_CUBLAS=1
export CUDA_DOCKER_ARCH=compute_75 # Tesla T4
export LLAMA_CUDA_NVCC=/usr/local/cuda/bin/nvcc
make
```

## Build Notes
```bash
# ggml-cuda.o
/usr/local/cuda/bin/nvcc --forward-unknown-to-host-compiler -use_fast_math -Wno-deprecated-gpu-targets -arch=compute_75 -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_MMV_Y=1 -DK_QUANTS_PER_ITERATION=2 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_K_QUANTS -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I/targets/x86_64-linux/include  -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -Wno-pedantic -Xcompiler -Wno-array-bounds -Wno-format-truncation -Wextra-semi -march=native -mtune=native -c ggml-cuda.cu -o ggml-cuda.o

# main
g++ -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_K_QUANTS -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I/targets/x86_64-linux/include  -std=c++11 -fPIC -O3 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread  -Wno-array-bounds -Wno-format-truncation -Wextra-semi -march=native -mtune=native  examples/main/main.cpp ggml.o llama.o common.o console.o grammar-parser.o k_quants.o ggml-cuda.o ggml-alloc.o -o main -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L/targets/x86_64-linux/lib
```


## Download GGUF model
```bash
cd models
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf
cd ..
```

## Run
```bash
# Load /etc/ld.so.conf.d/cuda-11-0.conf
ldconfig

./main -m models/llama-2-13b-chat.Q4_0.gguf -ngl 100 -p "What is a Large Language Model?"
```

## Debug
[VSCode launch.json](./.vscode/launch.json) 

# =====================================
# GDB
# =====================================
Run GDB command with `-exec` option in VSCode `DEBUG CONSOLE`:

### Print size of a variable
```bash
-exec p sizeof(ctx->kv)
```

### Examine a memory address
```bash
-exec p ctx
-exec x/72xb 0x4184c0
```

# =====================================
# Reading Code
# =====================================
Code flow:
![Code flow](./docs/images/llama.cpp.flow.png)

## &#9312; GGUF Model
Ref: [gguf.md](https://github.com/philpax/ggml/blob/gguf-spec/docs/gguf.md)  
Copy: [gguf.md](./docs/gguf.md)

## &#9313; Load Model
### Understanding `posix_memalign`

### /data00/home/son.nguyen/workspace/llama.cpp/common/common.cpp:802
```c++
llama_model * model  = llama_load_model_from_file(params.model.c_str(), mparams);
```

### /data00/home/son.nguyen/workspace/llama.cpp/llama.cpp:2516
```c++
llama_model_loader ml(fname, use_mmap);
```

### /data00/home/son.nguyen/workspace/llama.cpp/llama.cpp:1466
```c++
ctx_gguf = gguf_init_from_file(fname.c_str(), params);
```


## &#9314; Build Computational Graph
Replacements:
```
ggml_type_size\(([->_A-Za-z0-9]+)\)
type_traits[$1].type_size

ggml_element_size\(([->_A-Za-z0-9]+)\)
type_traits[$1->type].type_size

ggml_blck_size\(([->_A-Za-z0-9]+)\)
type_traits[$1].blck_size

ggml_is_quantized\(([->_A-Za-z0-9]+)\)
type_traits[$1].is_quantized

ggml_get_data\(([->_A-Za-z0-9]+)\)
$1->data
```

## &#9315; Execute Computational Graph

## &#9316; Kernel Launch
