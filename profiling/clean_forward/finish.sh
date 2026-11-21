set -eu
d=profiling/clean_forward
for pair in forest_run_113 forest_run_557 kitchen_cook_113 kitchen_cook_557 snow_ski_113 forest_run_917 snow_ski_917 bird_flight_919; do
    while ! test -s "$d/$pair.json"; do sleep 10; done
done
python3 -c "import sys; from pathlib import Path; sys.path.insert(0,'reproduction/vbench_current/history/conditional_budget_v1'); import calibrate; calibrate.d=Path('$d'); calibrate.main()"
for job in dense:scope intersection_fast:scope intersection_fast:after; do
    bash local_setup/run.sh "$d/video.py" "${job%:*}" "${job#*:}" > "$d/${job}.log" 2>&1
done
while ! test -s "$d/intersection_fast_before.mp4"; do sleep 10; done
bash local_setup/run.sh "$d/measure.py"
python3 "$d/board.py"
