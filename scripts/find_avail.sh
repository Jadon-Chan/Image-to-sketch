#!/bin/env bash

vrams=$(nvidia-smi --query-gpu=utilization.memory --format=csv | awk '{print $1}' | tail -n 8)

num=0
i=0
avail=-1

for vram in $vrams; do
	if [[ "$vram" == "0" ]]; then
		((num++))
		if [[ avail -eq -1 ]]; then
			avail=$i
		fi
	fi
	((i++))
done

echo $num
echo $avail
