#!/bin/bash
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
#python clean.py
rm *.aux
rm *.bbl *.blg *.log
