# Get the directory of the script
# Check if script_dir is provided as an argument, otherwise set it to the directory of the script

script_dir=$(dirname $0)
root=$(dirname $(dirname $0))
echo "script_dir: $script_dir"

python run_infer.py \
--gpu='0' \
--nr_types=3 \
--type_info_path=$script_dir/type_info.json \
--batch_size=7 \
--model_mode=fast \
--model_path=$script_dir/EMIP_withcontrastive_withoutremovingignoredarea.tar \
--nr_inference_workers=8 \
--nr_post_proc_workers=16 \
tile \
--input_dir=$root/test_data/output/patches_infer/ \
--output_dir=$root/test_data/output/output_infer/pred/ \
--mem_usage=0.1 \
--draw_dot \
--save_qupath

