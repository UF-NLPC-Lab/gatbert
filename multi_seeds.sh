#!/bin/bash

config=$1
outdir=$2
name=$3
n_seeds=${4:-10}

if [ ! -f $config ]
then
    echo Config $config does not exist
    exit 1
fi

mkdir -p $outdir
if [ ! -d $outdir ]
then
    echo Could not make $outdir
    exit 1
fi

for i in $(seq 1 $n_seeds)
do
    python -m gatbert.fit_and_test \
        -c $config \
        --seed_everything $i \
        --trainer.logger.init_args.save_dir $outdir \
        --trainer.logger.init_args.name $name \
        --trainer.logger.init_args.version seed_$i 
    echo $i
done
