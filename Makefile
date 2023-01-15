default: main

all: fftw_test main

main:
	gcc main.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O3 -o strings.out

fftw_test:
	gcc fftw_test.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O3 -o fftw_test.out
	./fftw_test.out
	python fftw_test.py

doc:
	pdflatex doc.tex

clean:
	rm -f *.out *.dat *.o *.pdf *.aux *.log *.xml *blx.bib


