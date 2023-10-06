main: common.h fundamentals.o cuda-stubs.o main.cpp
	$(CXX) -O0 -g -DGGML_USE_K_QUANTS -lpthread -o main fundamentals.o cuda-stubs.o main.cpp

fundamentals.o: fundamentals.h fundamentals.c
	$(CC) -O0 -g -DGGML_USE_K_QUANTS -c fundamentals.c

cuda-stubs.o: cuda-stubs.c
	$(CC) -O0 -g -c cuda-stubs.c

clean:
	rm -vf *.o main
