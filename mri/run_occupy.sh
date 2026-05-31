#!/bin/bash

# Counter for how many consecutive minutes GPU 7 is idle
idle_count=0

# Threshold for idleness in minutes
threshold=10

while true; do
    # Get the GPU 7 utilization using nvidia-smi
    util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sed -n '8p')

    echo "$(date): GPU 7 Utilization = $util%"

    if [ "$util" -eq 0 ]; then
        ((idle_count++))
        echo "GPU 7 has been idle for $idle_count minute(s)."
    else
        idle_count=0
    fi

    if [ "$idle_count" -ge "$threshold" ]; then
        echo "GPU 7 has been idle for $threshold minutes. Running occupy.py..."
        python occupy.py
        break
    fi

    sleep 60
done