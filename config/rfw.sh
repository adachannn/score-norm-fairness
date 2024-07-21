#!/bin/bash

./score_norm.py -stg train,test -m M1 -demo race -d rfw -p original -pr ./scorenorm_dataset_protocol/protocols/RFW -dr ./embedding -o ./output

./score_norm.py -stg train,test -m M1.1 -demo race -d rfw -p original -pr ./scorenorm_dataset_protocol/protocols/RFW -dr ./embedding -o ./output

./score_norm.py -stg train,test -m M1.2 -demo race -d rfw -p original -pr ./scorenorm_dataset_protocol/protocols/RFW -dr ./embedding -o ./output

./score_norm.py -stg train,test -m M2 -demo race -d rfw -p original -pr ./scorenorm_dataset_protocol/protocols/RFW -dr ./embedding -o ./output

./score_norm.py -stg train,test -m M2.1 -demo race -d rfw -p original -pr ./scorenorm_dataset_protocol/protocols/RFW -dr ./embedding -o ./output

./score_norm.py -stg train,test -m M2.2 -demo race -d rfw -p original -pr ./scorenorm_dataset_protocol/protocols/RFW -dr ./embedding -o ./output

./score_norm.py -stg train,test -m M3,M4,M5 -demo race -d rfw -p original -pr ./scorenorm_dataset_protocol/protocols/RFW -dr ./embedding -o ./output