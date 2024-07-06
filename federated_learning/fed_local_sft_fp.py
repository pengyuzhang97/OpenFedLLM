import torch
import copy
import math
from trl import SFTTrainer
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict

# ===== Packages from ZO-LLM trainer =====
import inspect
import math
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.func import functional_call, jvp
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
# Integrations must be imported before ML frameworks:
from transformers.integrations import hp_params
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    TrainOutput,
    has_length,
    ShardedDDPOption,
    speed_metrics,
    HPSearchBackend,
)
from transformers.utils import (
    WEIGHTS_NAME,
    is_apex_available,
    is_in_notebook,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)


def get_fed_local_sft_trainer_fp_hybrid(script_args, fed_args, model, tokenizer, training_args, local_train_dataset,
                              local_eval_dataset, formatting_prompts_func, data_collator, zo_eps):
    if (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer_FP(
            zo_eps=zo_eps,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_train_dataset,
            eval_dataset=local_eval_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer


def get_fed_local_sft_trainer_fp(script_args, fed_args, model, tokenizer, training_args, local_train_dataset,
                              local_eval_dataset, formatting_prompts_func, data_collator, global_dict, local_auxiliary,
                              global_auxiliary, zo_eps):
    if fed_args.fed_alg == 'fedprox':
        trainer = SFTTrainerFedProx_FP(
            zo_eps=zo_eps,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = SFTTrainerSCAFFOLD_FP(
            zo_eps=zo_eps,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = SFTTrainer_FP(
            zo_eps=zo_eps,
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=script_args.seq_length,
            train_dataset=local_train_dataset,
            eval_dataset=local_eval_dataset,
            formatting_func=formatting_prompts_func,
            data_collator=data_collator,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

# ===== Override ZO inner training loop =====
# ===== All wandbs are deleted. =====
logger = logging.get_logger(__name__)
class SFTTrainer_FP(SFTTrainer):
    def __init__(self, zo_eps, **kwargs):
        super(SFTTrainer_FP, self).__init__(**kwargs)
        self.zo_eps = zo_eps

    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """
        We overload the original training loop to add linear probing and MeZO. Search key word "MeZO added"
        for those updates.
        """
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
                self.sharded_ddp is not None
                and self.sharded_ddp != ShardedDDPOption.SIMPLE
                or is_sagemaker_mp_enabled()
                or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.


        # overload the optimizer here

        # 1.
        self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=0)

        # 2.
        # if args.trainer == "zo_adam":
        #     self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        #     # self.optimizer = {name: Adam([param], lr=args.learning_rate) for name, param in self.model.named_parameters()}
        #     # assert args.lr_scheduler
        #     assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        # elif args.trainer == "zo_sgd":
        #     self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)
        #     # self.optimizer = {name: SGD([param], lr=args.learning_rate) for name, param in self.model.named_parameters()}
        #     # print(f"### args.lr_scheduler: {args.lr_scheduler_type}")
        #     assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        # else:
        #     assert args.lr_scheduler_type == 'constant', "we did not implement lr_schedule."
        #     if args.optimizer == "adam":
        #         self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        #     elif args.optimizer == "sgd":
        #         self.optimizer = SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        # Main training loop
        total_steps = 0
        for epoch in range(epochs_trained, num_train_epochs):
            # print(f"-------------------------- Training Epoch {epoch} --------------------------")
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            # Start one epoch training
            for step, inputs in enumerate(epoch_iterator):

                total_steps += 1

                # torch.cuda.synchronize()
                step_start_time = time.time()

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                # MeZO added: estimate gradient
                tr_loss_step = self.zo_step(model, inputs)

                # sparse MeZo added
                # tr_loss_step = self.sparse_zo_step(model, inputs)

                if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        steps_in_epoch <= args.gradient_accumulation_steps
                        and (step + 1) == steps_in_epoch
                ):
                    # MeZO added: update model with the estimated gradient
                    # self.zo_update(model)

                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                # torch.cuda.synchronize()
                train_step_duration = time.time() - step_start_time

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                if self.args.eval_steps is not None and (total_steps + 1) % self.args.eval_steps == 0:
                    print(
                        f"=========================> Evaluating at step {total_steps + 1}... <=========================")
                    val_metrics = self.evaluate_func([], self.dev_samples)
                    test_metrics = self.evaluate_func([], self.eval_samples)
                    if "accuracy" in test_metrics:
                        self.log({"test_acc": test_metrics["accuracy"], "val_acc": val_metrics["accuracy"]})
                    else:
                        keys = list(test_metrics.keys())
                        log_dict = {}
                        for k in keys:
                            log_dict['test_' + k] = test_metrics[k]
                            log_dict['val_' + k] = val_metrics[k]
                        self.log(log_dict)

                # max_memory_allocated = 0
                # for device_id in range(torch.cuda.device_count()):
                #     # this is not accurate since max memory does not happen simultaneously across all devices
                #     max_memory_allocated += torch.cuda.max_memory_allocated(device_id)
                # self.log({"peak_mem": max_memory_allocated / 1024 ** 3,
                #           "step_consumption": train_step_duration * 1000})

            if step < 0:
                # Why would this happen? I don't know, but let's be safe.
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint.
        if self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    ############## MeZO ##############

    def zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for _, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            param.data = param.data + scaling_factor * z * self.zo_eps


    def sparse_zo_perturb_parameters(self, random_seed=None, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - random_seed: random seed for MeZO in-place perturbation (if it's None, we will use self.zo_random_seed)
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(random_seed if random_seed is not None else self.zo_random_seed)

        for _, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)

            threshold = torch.max(param.data.abs()).item() * self.frac
            mask = (param.data.abs()<=threshold)

            param.data = param.data + scaling_factor * z * self.zo_eps * mask

    def zo_forward(self, model, inputs):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        model.eval()

        with torch.inference_mode():
            inputs = self._prepare_inputs(inputs)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                # Warning: this is copied from the original Huggingface Trainer. Untested.
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        return loss.detach()

    @torch.no_grad()
    def zo_step(self, model, inputs, frac = 0.5):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        args = self.args

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # self.sparse_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self.zo_perturb_parameters(scaling_factor=1)
        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        # two side perturbation
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()

        # Reset model back to its parameters at start of step
        self.zo_perturb_parameters(scaling_factor=1)

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                            dtype=param.data.dtype)

            graddiff_times_z = self.projected_grad * z

            # # previous implementation
            # # no param.grad involved
            # param.data -= self._get_learning_rate() * self.projected_grad * z

            # param.grad += graddiff_times_z.detach()
            # more mem-efficient:
            # run optimizer.step here to avoid caching all grad.
            param.grad = graddiff_times_z

            self.optimizer.step()  # will only update grad that is not None.
            # param.data = param.data - graddiff_times_z
            param.grad = None  # avoid further update.

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1



    @torch.no_grad()
    def sparse_zo_step(self, model, inputs, frac = 0.5):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """

        self.frac = frac

        # What parameters to optimize
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
                # # TODO avoid init the memory for grad.
                # param.grad = torch.zeros_like(param.data)
                param.grad = None  # Make sure the grad is empty and will not be updated.

        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.randint(1000000000)

        # self.sparse_random_seed = np.random.randint(1000000000)

        # First function evaluation
        # self.sparse_zo_perturb_parameters(scaling_factor=1)
        self.zo_perturb_parameters(scaling_factor=1)

        loss1 = self.zo_forward(model, inputs)

        # Second function evaluation
        # two side perturbation
        # self.sparse_zo_perturb_parameters(scaling_factor=-2)
        self.zo_perturb_parameters(scaling_factor=-2)
        loss2 = self.zo_forward(model, inputs)
        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()

        # Reset model back to its parameters at start of step
        # self.sparse_zo_perturb_parameters(scaling_factor=1)
        self.zo_perturb_parameters(scaling_factor=1)

        # Set the random seed to ensure that we sample the same z for perturbation/update
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device,
                            dtype=param.data.dtype)


            threshold = torch.max(param.data.abs()).item() * self.frac
            mask = (param.data.abs()<=threshold)

            graddiff_times_z = self.projected_grad * z * mask

            # # previous implementation
            # # no param.grad involved
            # param.data -= self._get_learning_rate() * self.projected_grad * z

            # param.grad += graddiff_times_z.detach()
            # more mem-efficient:
            # run optimizer.step here to avoid caching all grad.
            param.grad = graddiff_times_z

            self.optimizer.step()  # will only update grad that is not None.
            # param.data = param.data - graddiff_times_z
            param.grad = None  # avoid further update.

        # No gradient accumulation support
        assert self.args.gradient_accumulation_steps == 1

        return loss1






    def zo_update(self, model):
        """
        Update the parameters with the estimated gradients.
        """
        # # Optimizer step
        # self.optimizer.step()
        # print(type(self.optimizer), self.optimizer)
        self.lr_scheduler.step()  # NOTE When we use own optimizer, this will no longer update the lr anymore.
        # self.optimizer.zero_grad()
        # model.zero_grad()

    @staticmethod
    @torch.no_grad()
    def functional_call_loss(params, names, buffers, model, batch):
        params = {k: v for k, v in zip(names, params)}
        outputs = functional_call(model, (params, buffers), tuple(), kwargs=batch)
        return outputs

    ############## Misc overload functions ##############

    def _set_signature_columns_if_needed(self):
        """
        We overload this function for non-differentiable objective training to pass "gold" -- the gold text for the task
        """
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns += ["gold"]

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        We overload this function to fix an FSDP saving bug (before fix, it will likely cause OOM)
        """

        if output_dir is None:
            output_dir = self.args.output_dir

        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif is_sagemaker_mp_enabled():
            # Calling the state_dict needs to be done on the wrapped model and on all processes.
            os.makedirs(output_dir, exist_ok=True)
            state_dict = self.model_wrapped.state_dict()
            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
            if IS_SAGEMAKER_MP_POST_1_10:
                # 'user_content.pt' indicates model state_dict saved with smp >= 1.10
                Path(os.path.join(output_dir, "user_content.pt")).touch()
        elif (
                ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
                or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
                or self.fsdp is not None
        ):
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
            full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            # Fix the FSDP loading bug
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                state_dict = self.model.state_dict()
            # state_dict = self.model.state_dict()

            if self.args.should_save:
                self._save(output_dir, state_dict=state_dict)
        elif self.deepspeed:
            # this takes care of everything as long as we aren't under zero3
            if self.args.should_save:
                self._save(output_dir)

            if is_deepspeed_zero3_enabled():
                # It's too complicated to try to override different places where the weights dump gets
                # saved, so since under zero3 the file is bogus, simply delete it. The user should
                # either user deepspeed checkpoint to resume or to recover full weights use
                # zero_to_fp32.py stored in the checkpoint.
                if self.args.should_save:
                    file = os.path.join(output_dir, WEIGHTS_NAME)
                    if os.path.isfile(file):
                        # logger.info(f"deepspeed zero3: removing {file}, see zero_to_fp32.py to recover weights")
                        os.remove(file)

                # now save the real model if stage3_gather_16bit_weights_on_model_save=True
                # if false it will not be saved.
                # This must be called on all ranks
                if not self.deepspeed.save_16bit_model(output_dir, WEIGHTS_NAME):
                    logger.warning(
                        "deepspeed.save_16bit_model didn't save the model, since"
                        " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead, use"
                        " zero_to_fp32.py to recover weights"
                    )
                    self.deepspeed.save_checkpoint(output_dir)

        elif self.args.should_save:
            self._save(output_dir)

        # Push to the Hub when `save_model` is called by the user.
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save")


class SFTTrainerFedProx_FP(SFTTrainer_FP):
    def __init__(self, global_state, prox_mu, zo_eps, **kwargs):
        super(SFTTrainerFedProx_FP, self).__init__(zo_eps=zo_eps, **kwargs)
        self.global_state = global_state
        self.mu = prox_mu

    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(SFTTrainerFedProx_FP, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")  # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss


class SFTTrainerSCAFFOLD_FP(SFTTrainer_FP):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, zo_eps, **kwargs):
        super(SFTTrainerSCAFFOLD_FP, self).__init__(zo_eps=zo_eps, **kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]

    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name] - param) / (
                                self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para


class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model

    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
        for name in model_para.keys():
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)