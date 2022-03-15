#!/usr/bin/env bash

for i in 1 2 4 8 16; do
    python3 train.py --reward_updates_per_itr $i --suffix ru_$i --sample_from_agent_buffer
done

# for i in 1 2 4 8 16; do
#     python3 train.py --agent_actions_per_demo_transition $i --suffix aa_$i -oa identity
# done