#!/bin/bash
while python scripts/sleeptill.py *.tex; do ./scripts/tex.sh; done
