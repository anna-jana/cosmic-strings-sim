default: doc

doc:
	pdflatex doc.tex # -synctex=1 -interaction=nonstopmode
	bibtex doc
	pdflatex doc.tex # -synctex=1 -interaction=nonstopmode

clean:
	rm -rf *.pdf *.aux *.log *.bbl *.blg *.xml *blx.bib


