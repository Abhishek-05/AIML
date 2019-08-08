#!/bin/sh
if [ "$#" -eq 2 ]; then
  python decoder_d.py $1 $2
fi
if [ "$#" -eq 3 ]; then
  python decoder_s.py $1 $2 $3
fi