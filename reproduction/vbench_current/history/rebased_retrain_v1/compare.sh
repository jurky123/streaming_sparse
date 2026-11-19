set -eu
cd /mnt/data/vbench_run_20261118/previews/rebased_retrain_v1
for scene in 0 1; do
 for idx in 11 21; do
  ffmpeg -v error -y -filter_complex_threads 2 -i "${scene}_11_dense.mp4" -i "${scene}_${idx}_original.mp4" -i "${scene}_${idx}_trained.mp4" -filter_complex "[0:v]scale=384:256,drawtext=text='Dense':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black[a];[1:v]scale=384:256,drawtext=text='Original':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black[b];[2:v]scale=384:256,drawtext=text='Retrained':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=black[c];[a][b][c]hstack=3" -c:v libx264 -threads 4 -crf 18 -preset veryfast -movflags +faststart "compare_${scene}_${idx}.mp4"
 done
done
for scene in 0 1; do
 ffmpeg -y -v error -filter_complex_threads 2 -i "${scene}_11_original.mp4" -i "${scene}_11_trained.mp4" -i "${scene}_21_original.mp4" -i "${scene}_21_trained.mp4" -filter_complex "[0:v]scale=576:384,drawtext=text='10% Original':expansion=none:x=10:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black[a];[1:v]scale=576:384,drawtext=text='10% Retrained':expansion=none:x=10:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black[b];[2:v]scale=576:384,drawtext=text='15% Original':expansion=none:x=10:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black[c];[3:v]scale=576:384,drawtext=text='15% Retrained':expansion=none:x=10:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black[d];[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0" -c:v libx264 -threads 4 -crf 18 -preset veryfast -movflags +faststart "side_by_side_${scene}.mp4"
done
