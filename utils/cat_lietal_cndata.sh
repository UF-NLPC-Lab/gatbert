#!/bin/bash

function gunzip_if_exists
{
	FILE_NAME=$1

	if [ ! -e $FILE_NAME ]
	then
		extended=${FILE_NAME}.gz
		if [ ! -e $extended ]
		then
			echo "Error: no $FILE_NAME or $extended"
			exit 1
		fi
		gunzip $extended
	fi
}
gunzip_if_exists train100k.txt || exit 1
gunzip_if_exists dev1.txt || exit 1
gunzip_if_exists dev2.txt || exit 1
cut -f 1-3 train100k.txt dev1.txt dev2.txt | sed 's/^/DUMMY\t/'
