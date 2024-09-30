
# About LoRA Learning

This project is an application of [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) to Stable Diffusion. It draws significant reference from [cloneofsimo's repository](https://github.com/cloneofsimo/lora), with deep appreciation for their work.

Typically, LoRA is applied to Linear and 1x1 Conv2d layers, but it can be extended to 3x3 Conv2d layers, as first demonstrated by [cloneofsimo](https://github.com/cloneofsimo/lora) and further validated by KohakuBlueleaf's [LoCon](https://github.com/KohakuBlueleaf/LoCon).

# Types of LoRA Supported

We support two unique types of LoRA, as named within this repository:

1. **LoRA-LierLa**: LoRA for Linear and 1x1 Conv2d layers.
2. **LoRA-C3Lier**: Extends support to 3x3 Conv2d layers.

LoRA-C3Lier may yield higher precision as it applies to more layers than LoRA-LierLa.

# Training LoRA Models

We use the `train_network.py` script for training. A higher learning rate, between `1e-4` to `1e-3`, is recommended for LoRA.

```bash
accelerate launch --num_cpu_threads_per_process 1 train_network.py        
      --pretrained_model_name_or_path=<path_to_pretrained_model>
      --dataset_config=<data_config>.toml     
      --output_dir=<output_directory>     
      --output_name=<output_name>     
      --save_model_as=safetensors     
      --prior_loss_weight=1.0     
      --max_train_steps=400     
      --learning_rate=1e-4     
      --optimizer_type="AdamW8bit"     
      --xformers     
      --mixed_precision="fp16"     
      --cache_latents     
      --gradient_checkpointing     
      --save_every_n_epochs=1     
      --network_module=networks.lora
```

For more details, please refer to the [common training document](./train_README-ja.md).

# DyLoRA Training

DyLoRA optimizes the rank dynamically during training, allowing the system to adapt the rank based on the dataset and tasks.

# Hierarchical Learning Rates

The hierarchical approach allows separate learning rates for different layers of the U-Net model, enhancing performance. 

```bash
--network_args "down_lr_weight=0.5,1.0" "mid_lr_weight=2.0" "up_lr_weight=1.5"
```

# Additional Scripts

Scripts like `merge_lora.py` and `svd_merge_lora.py` are available for merging models and applying LoRA models into Stable Diffusion models. For detailed usage instructions, refer to the project repository.

```
