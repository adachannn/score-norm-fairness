# The lists of methods that are evaluated
METHODS="M1 M1.1 M1.2 M2 M2.1 M2.2 M3 M4 M5"
# the list of extractors that are evaluated; remove E4 for the RFW dataset
EXTRACTORS="E1 E2 E3 E4 E5"

dataset="vgg2" # or rfw
protocol="short-demo" # or "random" or "original" for RFW dataset
demo="race" # or "gender" for VGG2 dataset
basedir="output" # Where your results are stored


# evaluate all extractors separately
for extractor in $EXTRACTORS; do
    # raw scores of this extractor
    FILES=$basedir/$dataset/$protocol/${demo}_${extractor}_raw.csv
    LABELS="raw"
    for method in $METHODS; do
        # normalized scores for the given method
        file="$basedir/$dataset/$protocol/${demo}_${extractor}_${method}_normed.csv"
        if [ -f $file ]; then
            FILES="$FILES $file"
            LABELS="$LABELS $method"
        fi
    done

    echo "Evaluating $extractor for demo $demo in directory $basedir/$dataset/$protocol"
    # compute true-match-rate table
    tmr-table -s $FILES -t $LABELS -T 3 -o $basedir/$dataset/$protocol/${demo}_${extractor}_table.json
    # plot the WERM report
    werm-report -d $dataset -s $FILES -t $LABELS -v $demo -o $basedir/$dataset/$protocol/${demo}_${extractor}_report.pdf
done
