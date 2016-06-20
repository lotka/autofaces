#!/bin/bash
pdflatex main.tex
bibtex report
pdflatex main.tex
pdflatex main.tex
#python clean.py
rm *.aux
rm *.bbl *.blg *.log
