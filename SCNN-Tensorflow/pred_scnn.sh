echo "Evaluating on the test set";
python -m lane-detection-model.tools.test_lanenet --weights_path ../model_culane-71-3/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000 --image_path /home/anelise/applied/lane_detection/dsets/culane/list/test.txt --save_dir output --image_bp /home/anelise/applied/lane_detection/dsets/culane/

#echo "Evaluating on the val set";
#python -m lane-detection-model.tools.test_lanenet --weights_path ../model_culane-71-3/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000 --image_path /home/anelise/applied/lane_detection/dsets/culane/list/val.txt --save_dir output --image_bp /home/anelise/applied/lane_detection/dsets/culane/
