#!/bin/bash
pdflatex -halt-on-error main.tex
bibtex main
pdflatex -halt-on-error main.tex
pdflatex -halt-on-error main.tex
mv main.pdf report.pdf
#python clean.py
rm -rf *.aux *.toc *.out *.bbl *.blg *.log *.brf
