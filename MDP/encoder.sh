#!/bin/sh
if [ "$#" -eq 1 ]; then
  python encoder_d.py $1
fi
if [ "$#" -eq 2 ]; then
  python encoder_s.py $1 $2
fi