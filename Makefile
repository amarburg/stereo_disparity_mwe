
smoothed.mov: output/smoothed_*.png
	ffmpeg -r 4 -i output/smoothed_%05d.png -pix_fmt yuv420p -vcodec prores -y $@

left.mov: output/left_disparity_*.png
	ffmpeg -r 4 -i output/left_disparity_%05d.png -pix_fmt yuv420p -vcodec prores -y $@
