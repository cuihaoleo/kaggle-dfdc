#!/bin/bash

export PYTHONPATH="external/Pytorch_Retinaface/$PYTHONPATH"

video_root="$1"
output_root="$2"

while read -r path; do
    name=$(basename "$path")
    name="${name%.*}"
    outdir="$output_root/$name"

    mkdir -p "$outdir"
    python3 make_dataset.py "$path" "$outdir"
done < <(find "$video_root" -type f -name '*.mp4')
