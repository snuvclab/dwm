
# # render object-only videos
# python data_processing/trumans/launch_blender_jobs_optimized.py --script-path data_processing/trumans/blender_ego_rgb_depth_optimized.py \
#  --save-path ./data/trumans/ego_render_fov90/ \
#  --only-object --video-output --no-depth --auto-split-clips --scenes trumans_all.txt\


# render interaction videos 16fps, 480p
# python data_processing/trumans/launch_blender_jobs_optimized.py --script-path data_processing/trumans/blender_ego_rgb_depth_optimized.py \
#  --save-path ./data/trumans/ego_render_fov90_fps16_480p/ \
#  --fps 16 --width 480 --height 480 \
#  --samples 64 \
#  --clip-length 81 \
#  --clip-stride 41 \
#  --frame-skip 2 \
#  --static --direct-clips \
#  --video-output --no-depth --scenes trumans_all.txt  

python data_processing/trumans/launch_blender_jobs_optimized.py --script-path data_processing/trumans/blender_ego_rgb_depth_optimized.py \
 --save-path ./data/trumans/ego_render_fov90_fps16_480_832/ \
 --fps 16 --width 832 --height 480 \
 --samples 64 \
 --clip-length 81 \
 --clip-stride 41 \
 --frame-skip 2 \
 --static --direct-clips \
 --video-output --no-depth --scenes trumans_all.txt  