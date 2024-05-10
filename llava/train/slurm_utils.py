import datetime
import logging
import logging.handlers
import os
import sys
import requests
import time
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
import torch
import warnings

def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_local_rank():
    if not torch.distributed.is_initialized():
        return 0
    num_gpus = torch.cuda.device_count()
    return get_rank() % num_gpus


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = None
        self.elapsed_time = 0

    def get_elapsed_time(self):
        if self.start_time is not None:
            return self.elapsed_time + (time.time() - self.start_time)
        
class TimeoutTerminateCallback(transformers.TrainerCallback):
    def __init__(self, args, total_time_limit=240, pre_terminate_time=10):
        self.training_args = args
        self.total_time_limit = total_time_limit
        self.pre_terminate_time = pre_terminate_time
        self.timer = Timer()
        self.timer.start()

        if args.local_rank == 0:
            print(f"Timer for terminate callback has been set.\nTotal limit: {total_time_limit}min\nPre terminate time: {pre_terminate_time}min")

        self.time_to_kill = (total_time_limit - pre_terminate_time) * 60


    def on_step_end(self, args, state, control, model, **kwargs):
        elapsed_time = self.timer.get_elapsed_time()
        
        if elapsed_time > self.time_to_kill:
            if args.local_rank == 0:
                print("Timeout, start to save checkpoint....")
            control.should_save = True
            control.should_training_stop = True

        return control