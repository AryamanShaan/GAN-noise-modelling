#!/bin/bash

# remote details
USER=shaana01
HOST=bigpurple.nyumc.org
REMOTE_DIR=/gpfs/home/shaana01/gan/wgan-gp_1/samples/

# local destination directory
LOCAL_DIR=/Users/shaan/Desktop/ML_MRI/code/GAN-noise-modelling/samples/

# remote files to download
FILES=(
  "sample_00.png"
  "sample_01.png"
  "sample_02.png"
  "sample_03.png"
  "sample_04.png"
  "sample_05.png"
  "sample_06.png"
  "sample_07.png"
  "sample_08.png"
  "sample_09.png"
)

# loop over files and download
for f in "${FILES[@]}"; do
    echo "Downloading $f ..."
    scp $USER@$HOST:$REMOTE_DIR/$f $LOCAL_DIR
done
