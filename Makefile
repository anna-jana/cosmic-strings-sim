default: main

main:
	gcc src/*.c -Wall -Wextra -std=c99 -pedantic -lm -lfftw3 -ggdb -O3 -o strings.out

check: main
	valgrind ./strings.out

clean:
	rm -rf *.out *.dat *.o *.hi __pycache__ *.pyc *.npy *.json

pyenv:
	conda create -n cosmic-strings python=3.10 numpy scipy matplotlib ipython pyfftw h5py numba

cdeps:
	sudo apt install gcc make valgrid gdb libfftw3-dev mpich


