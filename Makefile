default: main

all: fftw_test main doc

main:
	gcc *.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O0 -o strings.out

check: main
	valgrind ./strings.out

doc:
	pdflatex doc.tex
	bibtex doc
	pdflatex doc.tex

clean:
	rm -rf *.out *.dat *.o *.hi __pycache__ *.pyc *.pdf *.aux *.log *.bbl *.blg *.xml *blx.bib


