URL="https://osf.io/r6jc9/download"
DIR=`mktemp -d`
ZIP="$DIR/gtnet.deploy.zip"
OUTDIR="gtnet/models"

echo "Downloading model from $URL. Saving to $DIR"
curl -L -o $ZIP $URL
if [ ! -e $OUTDIR ]; then
    echo "Creating $OUTDIR"
    mkdir $OUTDIR
fi
echo "Extracting to $OUTDIR"
unzip -j -d $OUTDIR $ZIP
