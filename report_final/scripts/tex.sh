#!/bin/bash
rm bib/autofaces-*
rm bib/reading*
python scripts/bibclean.py bib/*.bib
pdflatex -halt-on-error main.tex
bibtex main
pdflatex -halt-on-error main.tex
pdflatex -halt-on-error main.tex
cp main.pdf report.pdf
#python clean.py
rm -rf *.aux *.toc *.out *.bbl *.blg *.log *.brf
