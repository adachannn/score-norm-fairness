# download protocol if not done yet
if [ ! -d protocol ]; then
  mkdir protocol
fi

cd protocol

wget -O protocols.tar.gz https://seafile.ifi.uzh.ch/f/c1623c5b26004f56b5ba/?dl=1


tar -xzvf protocols.tar.gz