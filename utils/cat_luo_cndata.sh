#!/bin/bash

echo "If you're reading this and the script hasn't terminated, you probably meant to run './cat_luo_cn_data.sh < conceptnet_english.txt'" >&2

cut -f3,1,2 /dev/stdin | awk '{printf "DUMMY\t/r/%s\t/c/en/%s\t/c/en/%s\n", $3, $1, $2}'

