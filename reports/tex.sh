#!/bin/bash
python clean.py
pdflatex report.tex
bibtex report
pdflatex report.tex
pdflatex report.tex
