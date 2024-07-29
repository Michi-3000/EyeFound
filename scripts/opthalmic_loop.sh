current_time=$(date +"%Y-%m-%d-%H%M")
for epoch in 33; do
  checkpoint="checkpoint-${epoch}.pth"
  for name in 'IDRiD' 'OCTID' 'PAPILA_v2' 'Retina' 'JSIEC' 'MESSIDOR2_v2' 'Aptos2019' 'Glaucoma_Fundus' 'OCTDL'; do
    CUDA_VISIBLE_DEVICES=2 python main_finetune_multi_randnloop.py \
      --now_epoch $epoch \
      --test_num 5 \
      --data_name $name \
      --batch_size 16 \
      --world_size 1 \
      --model vit_large_patch16 \
      --epochs 50 \
      --blr 5e-3 --layer_decay 0.65 \
      --weight_decay 0.05 --drop_path 0.2 \
      --output_dir "/home/danli/workspace/pretrain/mae_zwy/output_dir_downstream/all_dataset_$current_time" \
      --data_path /home/danli/data/public/MAE_process/ \
      --finetune "/home/danli/workspace/pretrain/mae_zwy/output_dir/2023-12-29-0017_retfound_cfp_0.8/$checkpoint" \
      --input_size 224
  done
done


