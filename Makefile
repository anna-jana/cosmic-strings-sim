default: main

main:
	gcc src/*.c -Wall -Wextra -std=c99 -pedantic -fopenmp -lgsl -lgslcblas -lm -lfftw3 -ggdb -O3 -o strings.out

debug:
	gcc src/*.c -Wall -Wextra -std=c99 -pedantic -fopenmp -lgsl -lgslcblas -lm -lfftw3 -ggdb -O0 -o strings.out
	OMP_NUM_THREADS=1 valgrind ./strings.out -- --log 2.5

clean:
	rm -rf run*_output

pyenv:
	conda create -n cosmic-strings python=3.10 numpy scipy matplotlib ipython numba

cdeps:
	sudo apt install gcc make valgrid gdb libfftw3-dev mpich libgsl-dev libgsl-dbg libgslcblas0


