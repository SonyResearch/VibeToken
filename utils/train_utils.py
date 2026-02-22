"""Training utils for VibeToken."""
import json
import os
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict
import random
import gc

from data import SimpleImageDataset, PretoeknizedDataSetJSONL, PretokenizedWebDataset
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from modeling.modules import EMAModel, ReconstructionLoss_Single_Stage, ReconstructionLoss_AE
from modeling.vibetoken_model import VibeTokenModel, PretrainedTokenizer
from modeling.vibetoken_ae_model import VibeTokenAEModel
from evaluator import VQGANEvaluator

from utils.viz_utils import make_viz_from_samples, make_viz_from_refinement_steps
from torchinfo import summary
import accelerate

def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_pretrained_tokenizer(config, accelerator=None):
    if config.model.vq_model.finetune_decoder:
        pretrianed_tokenizer = None
    else:
        pretrianed_tokenizer = PretrainedTokenizer(config.model.vq_model.pretrained_tokenizer_weight)
        if accelerator is not None:
            pretrianed_tokenizer.to(accelerator.device)
    return pretrianed_tokenizer


def create_model_and_loss_module(config, logger, accelerator,
                                 model_type="vibetoken"):
    """Creates model and loss module."""
    logger.info("Creating model and loss module.")
    if model_type == "vibetoken":
        if config.model.sub_model_type == "vibetoken":
            model_cls = VibeTokenModel
            loss_cls = ReconstructionLoss_Single_Stage
        elif config.model.sub_model_type == "vibetoken_ae":
            model_cls = VibeTokenAEModel
            loss_cls = ReconstructionLoss_AE
        else:
            raise ValueError(f"Unsupported sub_model_type {config.model.sub_model_type}")
    else:
        raise ValueError(f"Unsupported model_type {model_type}")
    model = model_cls(config)

    if config.experiment.get("init_weight", ""):
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        if config.model.vq_model.finetune_decoder:
            pretrained_tokenizer_weight = torch.load(
                config.model.vq_model.pretrained_tokenizer_weight, map_location="cpu"
            )
            pretrained_tokenizer_weight = {"pixel_" + k:v for k,v in pretrained_tokenizer_weight.items() if not "encoder." in k}
            model_weight.update(pretrained_tokenizer_weight)
        
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    loss_module = loss_cls(config=config) if loss_cls is not None else None

    if accelerator.is_main_process:
        if model_type in ["vibetoken"]:
            logger.info("VibeToken model summary not implemented yet.")
        else:
            raise NotImplementedError

    return model, ema_model, loss_module


def create_optimizer(config, logger, model, loss_module,
                     model_type="vibetoken", need_discrminator=True):
    """Creates optimizer for model and discriminator."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n or 'embed' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )

    if (config.model.vq_model.finetune_decoder or model_type == "vibetoken") and need_discrminator:
        discriminator_learning_rate = optimizer_config.discriminator_learning_rate
        discriminator_named_parameters = list(loss_module.named_parameters())
        discriminator_gain_or_bias_params = [p for n, p in discriminator_named_parameters if exclude(n, p) and p.requires_grad]
        discriminator_rest_params = [p for n, p in discriminator_named_parameters if include(n, p) and p.requires_grad]

        discriminator_optimizer = optimizer_cls(
            [
                {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
                {"params": discriminator_rest_params, "weight_decay": optimizer_config.weight_decay},
            ],
            lr=discriminator_learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2)
        )
    else:
        discriminator_optimizer = None

    assert discriminator_optimizer is not None, "Discriminator optimizer is None with condition values: {config.model.vq_model.finetune_decoder} {model_type} {need_discrminator}"

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """Creates learning rate scheduler for model and discriminator."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    if discriminator_optimizer is not None:
        discriminator_lr_scheduler = get_scheduler(
            config.lr_scheduler.scheduler,
            optimizer=discriminator_optimizer,
            num_training_steps=config.training.max_train_steps * accelerator.num_processes - config.losses.discriminator_start,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None
    return lr_scheduler, discriminator_lr_scheduler


def create_dataloader(config, logger, accelerator):
    """Creates data loader for training and testing."""
    logger.info("Creating dataloaders.")
    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    if dataset_config.get("pretokenization", "") and dataset_config.get("dataset_with_text_label", False) is True:
        dataset = PretokenizedWebDataset(
            train_shards_path=dataset_config.train_shards_path_or_url,
            eval_shards_path=dataset_config.eval_shards_path_or_url,
            num_train_examples=config.experiment.max_train_examples,
            per_gpu_batch_size=config.training.per_gpu_batch_size,
            global_batch_size=total_batch_size_without_accum,
            num_workers_per_gpu=dataset_config.num_workers_per_gpu,
            resize_shorter_edge=preproc_config.resize_shorter_edge,
            crop_size=preproc_config.crop_size,
            random_crop=preproc_config.random_crop,
            random_flip=preproc_config.random_flip,
            normalize_mean=preproc_config.normalize_mean,
            normalize_std=preproc_config.normalize_std,
            process_recap=preproc_config.get("preproc_recap", True),
            use_recap_prob=preproc_config.get("use_recap_prob", 0.95)
        )
        train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    elif dataset_config.get("pretokenization", "") and dataset_config.get("dataset_with_text_label", False) is False:
        dataset = SimpleImageDataset(
            train_shards_path=dataset_config.train_shards_path_or_url,
            eval_shards_path=dataset_config.eval_shards_path_or_url,
            num_train_examples=config.experiment.max_train_examples,
            per_gpu_batch_size=config.training.per_gpu_batch_size,
            global_batch_size=total_batch_size_without_accum,
            num_workers_per_gpu=dataset_config.num_workers_per_gpu,
            resize_shorter_edge=preproc_config.resize_shorter_edge,
            crop_size=preproc_config.crop_size,
            random_crop=preproc_config.random_crop,
            random_flip=preproc_config.random_flip,
            dataset_with_class_label=dataset_config.get("dataset_with_class_label", True),
            dataset_with_text_label=dataset_config.get("dataset_with_text_label", False),
            res_ratio_filtering=preproc_config.get("res_ratio_filtering", False),
            min_tokens=preproc_config.min_tokens,
            max_tokens=preproc_config.max_tokens,
        )
        train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader
    else:
        if dataset_config.get("pretokenization", ""):
            train_dataloader = DataLoader(
                PretoeknizedDataSetJSONL(dataset_config.pretokenization),
                batch_size=config.training.per_gpu_batch_size,
                shuffle=True, drop_last=True, pin_memory=True)
            train_dataloader.num_batches = math.ceil(
                config.experiment.max_train_examples / total_batch_size_without_accum)
    
    return train_dataloader, eval_dataloader


class LazyVQGANEvaluator:
    """A lazy-loading wrapper for VQGANEvaluator that delays inception model initialization."""
    
    def __init__(self, device, enable_rfid=True, enable_inception_score=True, 
                 enable_codebook_usage_measure=False, enable_codebook_entropy_measure=False,
                 num_codebook_entries=1024, accelerator=None):
        self._device = device
        self._enable_rfid = enable_rfid
        self._enable_inception_score = enable_inception_score
        self._enable_codebook_usage_measure = enable_codebook_usage_measure
        self._enable_codebook_entropy_measure = enable_codebook_entropy_measure
        self._num_codebook_entries = num_codebook_entries
        self._accelerator = accelerator
        self._evaluator = None
        self._initialized = False
        
    def _ensure_initialized(self):
        """Initialize the real evaluator only when needed."""
        if not self._initialized:
            if self._accelerator and self._accelerator.num_processes > 1:
                if self._accelerator.is_main_process:
                    try:
                        from evaluator.inception import get_inception_model
                        _ = get_inception_model()
                    except Exception as e:
                        print(f"Warning: Failed to pre-load inception model: {e}")
                
                if self._accelerator:
                    self._accelerator.wait_for_everyone()
            
            try:
                self._evaluator = VQGANEvaluator(
                    device=self._device,
                    enable_rfid=self._enable_rfid,
                    enable_inception_score=self._enable_inception_score,
                    enable_codebook_usage_measure=self._enable_codebook_usage_measure,
                    enable_codebook_entropy_measure=self._enable_codebook_entropy_measure,
                    num_codebook_entries=self._num_codebook_entries
                )
                self._initialized = True
            except Exception as e:
                print(f"Warning: Failed to create VQGANEvaluator, using dummy: {e}")
                class DummyEvaluator:
                    def reset_metrics(self): pass
                    def update(self, real_images, fake_images, codebook_indices=None): pass
                    def result(self): 
                        return {"InceptionScore": 0.0, "rFID": 0.0, "CodebookUsage": 0.0, "CodebookEntropy": 0.0}
                self._evaluator = DummyEvaluator()
                self._initialized = True
    
    def reset_metrics(self):
        self._ensure_initialized()
        return self._evaluator.reset_metrics()
    
    def update(self, real_images, fake_images, codebook_indices=None):
        self._ensure_initialized()
        return self._evaluator.update(real_images, fake_images, codebook_indices)
    
    def result(self):
        self._ensure_initialized()
        return self._evaluator.result()


def create_evaluator(config, logger, accelerator):
    """Creates evaluator."""
    logger.info("Creating evaluator.")
    
    quantize_mode = config.model.vq_model.get("quantize_mode", "vq")
    if quantize_mode in ["vq", "softvq", "mvq"]:
        evaluator = LazyVQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=True,
            enable_codebook_entropy_measure=True,
            num_codebook_entries=config.model.vq_model.codebook_size,
            accelerator=accelerator
        )
    elif quantize_mode in ["vae", "ae"]:
        evaluator = LazyVQGANEvaluator(
            device=accelerator.device,
            enable_rfid=True,
            enable_inception_score=True,
            enable_codebook_usage_measure=False,
            enable_codebook_entropy_measure=False,
            accelerator=accelerator
        )
    else:
        raise NotImplementedError
    
    logger.info("Lazy evaluator creation completed.")
    return evaluator


def auto_resume(config, logger, accelerator, ema_model,
                num_update_steps_per_epoch, strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            local_ckpt_list = list(glob.glob(os.path.join(
                config.experiment.output_dir, "checkpoint*")))
            logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        else:
            local_ckpt_list = []
        
        if accelerator.num_processes > 1:
            checkpoint_count = torch.tensor(len(local_ckpt_list), device=accelerator.device)
            accelerate.utils.broadcast(checkpoint_count, 0)
            
            if checkpoint_count > 0:
                if accelerator.is_main_process:
                    if len(local_ckpt_list) > 1:
                        fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                        checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
                    else:
                        checkpoint_paths = local_ckpt_list
                    latest_checkpoint = checkpoint_paths[0]
                else:
                    latest_checkpoint = ""
                
                if accelerator.is_main_process:
                    checkpoint_path_tensor = torch.tensor([ord(c) for c in latest_checkpoint], device=accelerator.device, dtype=torch.long)
                    path_length = torch.tensor(len(latest_checkpoint), device=accelerator.device)
                else:
                    path_length = torch.tensor(0, device=accelerator.device)
                
                accelerate.utils.broadcast(path_length, 0)
                
                if not accelerator.is_main_process:
                    checkpoint_path_tensor = torch.zeros(path_length.item(), device=accelerator.device, dtype=torch.long)
                
                accelerate.utils.broadcast(checkpoint_path_tensor, 0)
                
                if not accelerator.is_main_process:
                    latest_checkpoint = ''.join([chr(c.item()) for c in checkpoint_path_tensor])
                
                global_step = load_checkpoint(
                    Path(latest_checkpoint),
                    accelerator,
                    logger=logger,
                    strict=strict
                )
                if config.training.use_ema:
                    ema_model.set_step(global_step)
                first_epoch = global_step // num_update_steps_per_epoch
            else:
                logger.info("Training from scratch.")
        else:
            if len(local_ckpt_list) >= 1:
                if len(local_ckpt_list) > 1:
                    fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                    checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
                else:
                    checkpoint_paths = local_ckpt_list
                global_step = load_checkpoint(
                    Path(checkpoint_paths[0]),
                    accelerator,
                    logger=logger,
                    strict=strict
                )
                if config.training.use_ema:
                    ema_model.set_step(global_step)
                first_epoch = global_step // num_update_steps_per_epoch
            else:
                logger.info("Training from scratch.")
        
        accelerator.wait_for_everyone()
    return global_step, first_epoch


def train_one_epoch(config, logger, accelerator,
                    model, ema_model, loss_module,
                    optimizer, discriminator_optimizer,
                    lr_scheduler, discriminator_lr_scheduler,
                    train_dataloader, eval_dataloader,
                    evaluator,
                    global_step,
                    model_type="vibetoken",
                    clip_tokenizer=None,
                    clip_encoder=None,
                    pretrained_tokenizer=None):
    """One epoch training."""
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()

    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)
    for i, batch in enumerate(train_dataloader):
        model.train()
        if "image" in batch:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            if config.training.get("variable_resolution", False):
                any2any = config.training.variable_resolution.get("any2any", True)

                dims = config.training.variable_resolution.dim
                ratios = config.training.variable_resolution.ratio
                assert len(dims) == len(ratios), "dims and ratios must have the same length"
                input_res = tuple(random.choices(dims, weights=ratios, k=1)[0])
                
                if any2any:
                    output_res = tuple(random.choices(dims, weights=ratios, k=1)[0])
                else:
                    output_res = input_res
                
                images = torch.nn.functional.interpolate(images, size=output_res, mode="bilinear", align_corners=False)
                input_images = torch.nn.functional.interpolate(images, size=input_res, mode="bilinear", align_corners=False)
            else:
                input_images = images
                output_res = (None, None)

        fnames = batch["__key__"]
        data_time_meter.update(time.time() - end)

        if pretrained_tokenizer is not None:
            pretrained_tokenizer.eval()
            proxy_codes = pretrained_tokenizer.encode(images)
        else:
            proxy_codes = None

        with accelerator.accumulate([model, loss_module]):
            additional_args = {}
            if config.model.get("train_with_attention", False):
                additional_args["key_attention_mask"] = batch["attention_mask"].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            reconstructed_images, extra_results_dict = model(input_images, height=output_res[0], width=output_res[1], **additional_args)
            autoencoder_loss, loss_dict = loss_module(
                images,
                reconstructed_images,
                extra_results_dict,
                global_step,
                mode="generator",
            )

            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    gathered_tensor = accelerator.gather(v)
                    autoencoder_logs["train/" + k] = gathered_tensor.mean().item()
                    del gathered_tensor
            
            torch.cuda.empty_cache()
            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if (config.model.vq_model.finetune_decoder or model_type == "vibetoken") and accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step=global_step,
                    mode="discriminator",
                )

                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                            gathered_tensor = accelerator.gather(v)
                            discriminator_logs["train/" + k] = gathered_tensor.mean().item()
                            del gathered_tensor

                torch.cuda.empty_cache()
                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()
        
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)
                
                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                    f"Recon Loss: {autoencoder_logs['train/reconstruction_loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                del autoencoder_logs, discriminator_logs, logs
                gc.collect()

                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0:
                if accelerator.is_main_process:
                    if config.training.get("use_ema", False):
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())

                    reconstruct_images(
                        model,
                        images[:config.training.num_generated_images],
                        fnames[:config.training.num_generated_images],
                        accelerator,
                        global_step + 1,
                        config.experiment.output_dir,
                        logger=logger,
                        config=config,
                        pretrained_tokenizer=pretrained_tokenizer
                    )

                    if config.model.sub_model_type == "vibetoken_ae":
                        generate_ae_images(
                            model, accelerator, global_step + 1,
                            config.experiment.output_dir, logger, config,
                            num_images=config.training.num_generated_images,
                        )

                    if config.training.get("use_ema", False):
                        ema_model.restore(model.parameters())
                
                accelerator.wait_for_everyone()


            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    eval_scores = eval_reconstruction(
                        config,
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        pretrained_tokenizer=pretrained_tokenizer
                    )
                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        ema_model.restore(model.parameters())
                else:
                    eval_scores = eval_reconstruction(
                        config,
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                        pretrained_tokenizer=pretrained_tokenizer
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break


    return global_step


@torch.no_grad()
def eval_reconstruction(
    config,
    model,
    eval_loader,
    accelerator,
    evaluator,
    pretrained_tokenizer=None
):
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    accelerator.wait_for_everyone()
    
    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )

        original_images = torch.clone(images)
        additional_args = {}
        if config.model.get("eval_with_attention", False):
            additional_args["key_attention_mask"] = batch["attention_mask"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        reconstructed_images, model_dict = local_model(images, **additional_args)

        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        reconstructed_images = torch.round(reconstructed_images * 255.0) / 255.0
        original_images = torch.clamp(original_images, 0.0, 1.0)
        
        if isinstance(model_dict, dict): 
            evaluator.update(original_images, reconstructed_images.squeeze(2), model_dict.get("min_encoding_indices", None))
        else:
            evaluator.update(original_images, reconstructed_images.squeeze(2), None)
    
    accelerator.wait_for_everyone()
    
    local_results = evaluator.result()
    
    if accelerator.num_processes > 1:
        gathered_results = {}
        for key, value in local_results.items():
            if isinstance(value, (int, float)):
                value_tensor = torch.tensor(value, device=accelerator.device)
                gathered_values = accelerator.gather(value_tensor)
                gathered_results[key] = gathered_values.mean().item()
            else:
                gathered_results[key] = value
        
        accelerator.wait_for_everyone()
        model.train()
        return gathered_results
    else:
        model.train()
        return local_results


@torch.no_grad()
def reconstruct_images(model, original_images, fnames, accelerator, 
                    global_step, output_dir, logger, config=None,
                    pretrained_tokenizer=None):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    _, _, height, width = original_images.shape
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        enc_tokens, encoder_dict = accelerator.unwrap_model(model).encode(original_images)
        reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens, height=height, width=width)
        if pretrained_tokenizer is not None:
            reconstructed_images = pretrained_tokenizer.decode(reconstructed_images.argmax(1))

    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images,
        reconstructed_images
    )
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_saving):
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()


@torch.no_grad()
def generate_ae_images(model, accelerator, global_step, output_dir, logger,
                       config, num_images=4):
    """Generate one-shot and iterative-refinement images for VibeTokenAE."""
    logger.info("Generating AE images (one-shot + refinement)...")
    local_model = accelerator.unwrap_model(model)
    local_model.eval()

    height = config.dataset.preprocessing.crop_size
    width = config.dataset.preprocessing.crop_size

    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    root_oneshot = Path(output_dir) / "gen_oneshot"
    root_refine = Path(output_dir) / "gen_refinement"
    os.makedirs(root_oneshot, exist_ok=True)
    os.makedirs(root_refine, exist_ok=True)

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        # One-shot generation
        oneshot_steps = local_model.generate(
            num_images, height, width, num_steps=1)
        # Iterative refinement (4 steps)
        refine_steps = local_model.generate(
            num_images, height, width, num_steps=4, refine_noise_deg=5.0)

    # Save one-shot grid
    oneshot_img, oneshot_log = make_viz_from_refinement_steps(oneshot_steps)
    oneshot_img.save(root_oneshot / f"{global_step:08d}_oneshot.png")

    # Save refinement grid
    refine_img, refine_log = make_viz_from_refinement_steps(refine_steps)
    refine_img.save(root_refine / f"{global_step:08d}_refinement.png")

    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {"AE One-Shot Generation": [oneshot_img]}, step=global_step)
        accelerator.get_tracker("wandb").log_images(
            {"AE Refinement Generation": [refine_img]}, step=global_step)
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"AE One-Shot Generation": oneshot_log[None]}, step=global_step)
        accelerator.get_tracker("tensorboard").log_images(
            {"AE Refinement Generation": refine_log[None]}, step=global_step)

    model.train()


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)
