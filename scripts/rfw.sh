#!/bin/bash

# original protocol
# score-norm -m M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5 -e E1 E2 E3 E5 -n race -d rfw -p original -P ./protocol/protocols/RFW -D ./embedding -o ./output
score-norm -m M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5 -e E6 -n race -d rfw -p original -P /local/scratch/yuinkwan/bias-mitigate/score-norm-fairness/protocol/protocols/RFW -D /local/scratch/yuinkwan/bias-mitigate/score-norm-fairness/embedding -o /local/scratch/yuinkwan/bias-mitigate/score-norm-fairness/output/20250527

# random protocol
# score-norm -m M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5 -e E1 E2 E3 E5 -n race -d rfw -p random -P ./protocol/protocols/RFW -D ./embedding -o ./output
