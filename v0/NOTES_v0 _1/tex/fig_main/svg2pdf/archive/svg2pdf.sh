#!/bin/bash

for file in ../svg/*.svg
    do
        filename=$(basename "$file")
        inkscape "$file" -d 1200 -A "${filename%.svg}.pdf" 
    done
