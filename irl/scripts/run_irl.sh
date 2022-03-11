#!/usr/bin/env bash

for i in 1 2 4 8 16; do
    python3 train.py --reward_updates_per_itr $i --suffix ru_$i -oa identity
done
