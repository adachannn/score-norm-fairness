# the list of extractors that are evaluated; add E4 for the VGG2 dataset
EXTRACTORS="E1 E2 E3 E5"

dataset="rfw" # or vgg2
protocol="original" # or "random" for RFW dataset or "short-demo" for VGG2 dataset
demo="race" # or "race"/"gender" for VGG2 dataset
basedir="output" # Where your results are stored
target_threshold=3 # or 2, 1

echo "Visualization of $dataset for demo $demo in directory $basedir/$dataset/$protocol"
visualization -s $basedir/$dataset/$protocol -n $demo -e $EXTRACTORS -T $target_threshold -o $basedir/$dataset/$protocol/${demo}_analysis_table.csv