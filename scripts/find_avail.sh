#!/bin/env bash

vrams=$(hy-smi | head -n 12 | tail -n 8 | awk '{print $6}')

num=0
i=0
avail=-1

for vram in $vrams; do
	if [[ "$vram" == "0%" ]]; then
		((num++))
		if [[ avail -eq -1 ]]; then
			avail=$i
		fi
	fi
	((i++))
done

echo $num
echo $avail
