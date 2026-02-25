"""Training script for VibeToken.

Reference:
    https://github.com/huggingface/open-muse
"""
import math
import os
import sys
from pathlib import Path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(parent_dir)

from accelerate.utils import set_seed
from accelerate import Accelerator

import torch
import wandb
from omegaconf import OmegaConf
from utils.logger import setup_logger

from utils.train_utils import (
    get_config, create_pretrained_tokenizer, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler, create_dataloader,
    create_evaluator, auto_resume, save_checkpoint, 
    train_one_epoch)


def main():
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")

    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="VibeToken", log_level="INFO",
     output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    if accelerator.is_main_process:
        if config.training.enable_wandb:
            wandb_config = config.training.get("wandb", {})
            wandb_project = wandb_config.get("project", config.experiment.project)
            wandb_entity = wandb_config.get("entity", None)
            wandb_name = wandb_config.get("name", config.experiment.name)
            wandb_tags = list(wandb_config.get("tags", []))
            wandb_notes = wandb_config.get("notes", None)
            wandb_resume_id = wandb_config.get("resume_id", None)

            wandb_init_kwargs = {
                "wandb": {
                    "name": wandb_name,
                    "dir": output_dir,
                    "resume": "allow",
                }
            }
            if wandb_entity:
                wandb_init_kwargs["wandb"]["entity"] = wandb_entity
            if wandb_tags:
                wandb_init_kwargs["wandb"]["tags"] = wandb_tags
            if wandb_notes:
                wandb_init_kwargs["wandb"]["notes"] = wandb_notes
            if wandb_resume_id:
                wandb_init_kwargs["wandb"]["id"] = wandb_resume_id

            accelerator.init_trackers(
                project_name=wandb_project,
                config=OmegaConf.to_container(config, resolve=True),
                init_kwargs=wandb_init_kwargs,
            )
            logger.info(f"WandB initialized - Project: {wandb_project}, Name: {wandb_name}")
        else:
            accelerator.init_trackers(config.experiment.name)

        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    accelerator.wait_for_everyone()

    # Create pretrained tokenizer in a synchronized manner
    if config.model.vq_model.is_legacy:
        if accelerator.is_main_process:
            logger.info("Creating pretrained tokenizer on main process...")
        accelerator.wait_for_everyone()
        pretrained_tokenizer = create_pretrained_tokenizer(config, accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info("Pretrained tokenizer creation completed.")
    else:
        pretrained_tokenizer = None

    if accelerator.is_main_process:
        logger.info("Creating model and loss module...")
    accelerator.wait_for_everyone()
    
    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type="vibetoken")
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Model creation completed.")

    optimizer, discriminator_optimizer = create_optimizer(config, logger, model, loss_module, model_type="vibetoken")

    lr_scheduler, discriminator_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer)

    if accelerator.is_main_process:
        logger.info("Creating dataloaders...")
    train_dataloader, eval_dataloader = create_dataloader(config, logger, accelerator)
    accelerator.wait_for_everyone()

    # Set up evaluator.
    if accelerator.is_main_process:
        logger.info("Setting up evaluator...")
    evaluator = create_evaluator(config, logger, accelerator)

    # Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    if config.model.vq_model.is_legacy:
        if config.model.vq_model.finetune_decoder:
            model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler = accelerator.prepare(
                model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler
            )
        else:
            model, optimizer, lr_scheduler = accelerator.prepare(
                model, optimizer, lr_scheduler
            )
    else:
        model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler = accelerator.prepare(
            model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler
        )

    if config.training.use_ema:
        ema_model.to(accelerator.device)

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum)
    num_update_steps_per_epoch = math.ceil(num_batches / config.training.gradient_accumulation_steps)
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    global_step = 0
    first_epoch = 0

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model, num_update_steps_per_epoch,
        strict=True)

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch(config, logger, accelerator,
                            model, ema_model, loss_module,
                            optimizer, discriminator_optimizer,
                            lr_scheduler, discriminator_lr_scheduler,
                            train_dataloader, eval_dataloader,
                            evaluator,
                            global_step,
                            pretrained_tokenizer=pretrained_tokenizer,
                            model_type="vibetoken")
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)

    if accelerator.is_main_process and config.training.enable_wandb:
        wandb.finish()
        logger.info("WandB run finished")
    accelerator.end_training()


if __name__ == "__main__":
    main()