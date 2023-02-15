default: main

main:
	gcc src/*.c -Wall -Wextra -std=c99 -pedantic -lm -lfftw3 -ggdb -O3 -o strings.out

check: main
	valgrind ./strings.out

clean:
	rm run*_output -rf

pyenv:
	conda create -n cosmic-strings python=3.10 numpy scipy matplotlib ipython numba

cdeps:
	sudo apt install gcc make valgrid gdb libfftw3-dev mpich


