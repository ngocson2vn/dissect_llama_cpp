main: common.h fundamentals.o ggml-cuda.o main.cpp
	$(CXX) -O0 -g -DGGML_USE_K_QUANTS -DGGML_USE_CUBLAS -lpthread -lcublas -lcudart -lpthread -L/usr/local/cuda/lib64 -o main fundamentals.o ggml-cuda.o main.cpp

fundamentals.o: fundamentals.h fundamentals.c
	$(CC) -O0 -g -DGGML_USE_K_QUANTS -c fundamentals.c

cuda-stubs.o: cuda-stubs.c
	$(CC) -O0 -g -c cuda-stubs.c

ggml-cuda.o: ggml-cuda.h ggml-cuda.cu
	/usr/local/cuda/bin/nvcc --forward-unknown-to-host-compiler -use_fast_math -Wno-deprecated-gpu-targets -arch=compute_75 -DGGML_CUDA_DMMV_X=32 -DGGML_CUDA_MMV_Y=1 -DK_QUANTS_PER_ITERATION=2 -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 -I. -Icommon -D_XOPEN_SOURCE=600 -D_GNU_SOURCE -DNDEBUG -DGGML_USE_K_QUANTS -DGGML_USE_CUBLAS -I/usr/local/cuda/include -I/opt/cuda/include -I/targets/x86_64-linux/include -std=c++11 -fPIC -O0 -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wmissing-declarations -Wmissing-noreturn -pthread -Wno-pedantic -Xcompiler "-Wno-array-bounds -Wno-format-truncation -Wextra-semi -march=native -mtune=native" -c ggml-cuda.cu -o ggml-cuda.o

clean:
	rm -vf *.o main
