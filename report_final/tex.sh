#!/bin/bash
pdflatex -halt-on-error main.tex
bibtex main
pdflatex -halt-on-error main.tex
pdflatex -halt-on-error main.tex
#python clean.py
rm -rf *.aux *.toc *.out *.bbl *.blg *.log
