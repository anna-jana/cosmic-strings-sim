default:
	gcc main.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O3 -o strings.out

fftw_test:
	gcc fftw_test.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O3 -o fftw_test.out
	./fftw_test.out
	python fftw_test.py

clean:
	rm *.out *.dat *.o

