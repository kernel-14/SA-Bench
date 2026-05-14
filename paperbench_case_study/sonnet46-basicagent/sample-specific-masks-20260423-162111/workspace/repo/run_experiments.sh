#!/bin/bash
# Run all experiments from the paper
# Tables 1, 2, 3 and Figure 4 (patch size ablation)

DATA_ROOT="./data"
OUTPUT_DIR="./outputs"
SEEDS="42 43 44"  # 3 seeds as in paper

DATASETS="CIFAR10 CIFAR100 SVHN GTSRB Flowers102 DTD UCF101 Food101 SUN397 EuroSAT OxfordPets"
METHODS_RESNET="pad narrow medium full smm"
METHODS_ABLATION="only_delta only_fmask single_channel smm"

# Table 1: ResNet-18 and ResNet-50 experiments
for MODEL in ResNet18 ResNet50; do
    for DATASET in $DATASETS; do
        for METHOD in $METHODS_RESNET; do
            for SEED in $SEEDS; do
                python src/train.py \
                    --dataset $DATASET \
                    --model $MODEL \
                    --method $METHOD \
                    --label_mapping ilm \
                    --epochs 200 \
                    --data_root $DATA_ROOT \
                    --output_dir $OUTPUT_DIR/${MODEL}/${DATASET}/${METHOD}/seed${SEED} \
                    --seed $SEED
            done
        done
    done
done

# Table 2: ViT-B32 experiments
for DATASET in $DATASETS; do
    for METHOD in $METHODS_RESNET; do
        for SEED in $SEEDS; do
            python src/train.py \
                --dataset $DATASET \
                --model ViT_B32 \
                --method $METHOD \
                --label_mapping ilm \
                --epochs 200 \
                --data_root $DATA_ROOT \
                --output_dir $OUTPUT_DIR/ViT_B32/${DATASET}/${METHOD}/seed${SEED} \
                --seed $SEED
        done
    done
done

# Table 3: Ablation studies (ResNet-18)
for DATASET in $DATASETS; do
    for METHOD in $METHODS_ABLATION; do
        for SEED in $SEEDS; do
            python src/train.py \
                --dataset $DATASET \
                --model ResNet18 \
                --method $METHOD \
                --label_mapping ilm \
                --epochs 200 \
                --data_root $DATA_ROOT \
                --output_dir $OUTPUT_DIR/ablation/${DATASET}/${METHOD}/seed${SEED} \
                --seed $SEED
        done
    done
done

# Figure 4: Patch size ablation (ResNet-18)
for DATASET in $DATASETS; do
    for PATCH_SIZE in 1 2 4 8 16; do
        for SEED in $SEEDS; do
            python src/train.py \
                --dataset $DATASET \
                --model ResNet18 \
                --method smm \
                --patch_size $PATCH_SIZE \
                --label_mapping ilm \
                --epochs 200 \
                --data_root $DATA_ROOT \
                --output_dir $OUTPUT_DIR/patch_size/${DATASET}/patch${PATCH_SIZE}/seed${SEED} \
                --seed $SEED
        done
    done
done

# Table 10: Different label mapping methods (ResNet-18)
for DATASET in $DATASETS; do
    for LM in ilm flm rlm; do
        for METHOD in full smm; do
            for SEED in $SEEDS; do
                python src/train.py \
                    --dataset $DATASET \
                    --model ResNet18 \
                    --method $METHOD \
                    --label_mapping $LM \
                    --epochs 200 \
                    --data_root $DATA_ROOT \
                    --output_dir $OUTPUT_DIR/label_mapping/${DATASET}/${LM}/${METHOD}/seed${SEED} \
                    --seed $SEED
            done
        done
    done
done

echo "All experiments completed!"
