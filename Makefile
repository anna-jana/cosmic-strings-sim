default:
	gcc main.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O3 -o strings.out

test:
	gcc test.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O3 -o test.out
	./test.out
	python test.py

clean:
	rm *.out *.dat *.o

