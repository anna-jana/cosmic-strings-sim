default: main

all: fftw_test main doc

main:
	gcc *.c -Wall -std=c99 -pedantic -lm -lfftw3 -g -O3 -o strings.out

check: main
	valgrind ./strings.out

doc:
	pdflatex doc.tex # -synctex=1 -interaction=nonstopmode
	bibtex doc
	pdflatex doc.tex # -synctex=1 -interaction=nonstopmode

clean:
	rm -rf *.out *.dat *.o *.hi __pycache__ *.pyc *.pdf *.aux *.log *.bbl *.blg *.xml *blx.bib

pyenv:
	conda create -n cosmic-strings python=3.10 numpy scipy matplotlib ipython pyfftw h5py

cdeps:
	sudo apt install gcc make valgrid gdb libfftw3-dev

analysis: pyenv
	python analysis.py

