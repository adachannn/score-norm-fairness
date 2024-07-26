#!/bin/bash

# gender demo
score-norm -m M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5 -e E1 E2 E3 E4 E5 -n gender -d vgg2 -p short-demo -P ./protocol/protocols/VGGFace2 -D ./embedding -o ./output

# race demo
score-norm -m M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5 -e E1 E2 E3 E4 E5 -n race -d vgg2 -p short-demo -P ./protocol/protocols/VGGFace2 -D ./embedding -o ./output
