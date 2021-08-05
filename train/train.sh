#!/bin/bash

python train.py --cuda --gpu 0 --epoches 2 --batchSize 4 --lr 1e-4 \
--dynamic_filter --both_sty_con --relax_style \
--style_content_loss --recon_loss --tv_loss --temporal_loss \
--data_sigma --data_w --adaversarial_loss \
--content_data data/content/val2017 \
--style_data ../../datasets/animal_textures \
--outf result/animal_textures \
--loadSize 512

python train.py --cuda --gpu 0 --epoches 2 --batchSize 4 --lr 1e-4 \
--dynamic_filter --both_sty_con --relax_style \
--style_content_loss --recon_loss --tv_loss --temporal_loss \
--data_sigma --data_w --adaversarial_loss \
--content_data data/content/val2017 \
--style_data ../../datasets/animal_drums \
--outf result/animal_drums \
--loadSize 512

python train.py --cuda --gpu 0 --epoches 2 --batchSize 4 --lr 1e-4 \
--dynamic_filter --both_sty_con --relax_style \
--style_content_loss --recon_loss --tv_loss --temporal_loss \
--data_sigma --data_w --adaversarial_loss \
--content_data data/content/val2017 \
--style_data ../../datasets/mason/irises \
--outf result/irises \
--loadSize 512



