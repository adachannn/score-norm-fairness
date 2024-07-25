#!/usr/bin/bash

if [ ! -d features ]; then
  mkdir features
fi

cd features
wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FRFW_features%2FRFW_E1.tar.gz&dl=1
wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FRFW_features%2FRFW_E2.tar.gz&dl=1
wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FRFW_features%2FRFW_E3.tar.gz&dl=1
wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FRFW_features%2FRFW_E5.tar.gz&dl=1

wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FVGG2_features%2FVGG2_short_E1.tar.gz&dl=1
wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FVGG2_features%2FVGG2_short_E2.tar.gz&dl=1
wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FVGG2_features%2FVGG2_short_E3.tar.gz&dl=1
wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FVGG2_features%2FVGG2_short_E4.tar.gz&dl=1
wget https://seafile.ifi.uzh.ch/d/46473c78a796425a8022/files/?p=%2Fpretrained_features%2FVGG2_features%2FVGG2_short_E5.tar.gz&dl=1


for f in "*.tar.gz"; do
    echo tar -xzf $f;
done
