current_time=$(date +"%Y-%m-%d-%H%M")
# for epoch in {0..16..1}; do
for epoch in 33; do
  checkpoint="checkpoint-${epoch}.pth"
  for name in 'MESSIDOR2_v2' 'IDRiD' 'Aptos2019'; do
    CUDA_VISIBLE_DEVICES=3 python main_finetune_multi_cross.py \
      --now_epoch $epoch \
      --test_num 1 \
      --data_name $name \
      --batch_size 16 \
      --world_size 1 \
      --model vit_large_patch16 \
      --epochs 50 \
      --blr 5e-3 --layer_decay 0.65 \
      --weight_decay 0.05 --drop_path 0.2 \
      --output_dir "/home/danli/workspace/pretrain/mae_zwy/output_dir_downstream/crossdata_$current_time" \
      --test_dir "/home/danli/workspace/pretrain/mae_zwy/output_dir_downstream/all_dataset_2024-02-18-0013_testtimes5" \
      --data_path /home/danli/data/public/MAE_process/ \
      --finetune "/home/danli/workspace/pretrain/mae_zwy/output_dir/2023-12-29-0017_retfound_cfp_0.8/$checkpoint" \

      --input_size 224
  done
done


#'JSIEC' 'PAPILA' 'IDRiD' 'MESSIDOR2' 'Aptos2019' 'Retina' 'Glaucoma_Fundus' 'OCTID'
#{5..16..1}
#--output_dir "/home/danli/workspace/pretrain/mae_zwy/output_dir_downstream/all_dataset_$current_time" \
#/home/danli/workspace/pretrain/mae_zwy/output_dir/2023-12-27-1933_retfound_cfp
#/
# /home/danli/workspace/pretrain/mae_zwy/output_dir/2023-12-27-1933_retfound_cfp