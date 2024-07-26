#!/bin/bash

# original protocol
score-norm -m M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5 -e E1 E2 E3 E5 -n race -d rfw -p original -P ./protocol/protocols/RFW -D ./embedding -o ./output

# random protocol
score-norm -m M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5 -e E1 E2 E3 E5 -n race -d rfw -p random -P ./protocol/protocols/RFW -D ./embedding -o ./output
