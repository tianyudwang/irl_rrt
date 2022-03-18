#!/usr/bin/env bash

for i in 1 2 4 8 16; do
    python3 train_irl.py --n_reward_updates_per_itr $i --suffix ru_$i --agent_action_from_demo_state
done
