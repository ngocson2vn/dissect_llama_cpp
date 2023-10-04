main: common.h fundamentals.o main.cpp
	$(CXX) -O0 -g -DGGML_USE_K_QUANTS -o main fundamentals.o main.cpp

fundamentals.o: fundamentals.h fundamentals.c
	$(CC) -O0 -g -DGGML_USE_K_QUANTS -c fundamentals.c

clean:
	rm -vf *.o main
