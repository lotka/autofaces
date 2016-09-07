#!/bin/bash
./scripts/tex.sh
while python scripts/sleeptill.py *.tex; do ./scripts/tex.sh; done
./scripts/autocompile.sh
