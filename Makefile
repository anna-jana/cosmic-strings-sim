default: main

all: fftw_test main doc

main:
	gcc propagator.c util.c main.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O0 -o strings.out

check: main
	valgrind ./strings.out

fftw_test:
	gcc fftw_test.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O3 -o fftw_test.out
	./fftw_test.out
	python fftw_test.py

doc:
	pdflatex doc.tex
	bibtex doc
	pdflatex doc.tex

clean:
	rm -rf *.out *.dat *.o *.hi __pycache__ *.pyc *.pdf *.aux *.log *.bbl *.blg *.xml *blx.bib


