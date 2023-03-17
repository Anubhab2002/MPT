# basic imports of logging, os, sys, numpy, typing(helps and aliases with typing at runtime)
import logging
import os
import sys
import numpy as np
from typing import Dict

# import torch, datasets, transformers and required methods from transformers library. Datasets and transformers both from huggingface
import torch
import datasets
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

# import args
from arguments import get_args

# get the utils files --> READ task.utils
from tasks.utils import *


logger = logging.getLogger(__name__)

# the training function 
def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None # initially set the trainer checkpoint to none
    if resume_from_checkpoint is not None: # basically if a checkpoint exists then take that up as the checkpoint for the training
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None: # set the checkpoint to last checkpoint --> IDK THE DIFFERENCE BETWEEN THE 2
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint) # call the train function from the trainer using the checkpoint available
    # trainer.save_model()

    # get the metrics for the training and save them and log them
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    trainer.log_best_metrics() # log the best metrics


# function to perform evaluation on the test dataset...takes the trainer as the parameter though (I think this is for validation)
def evaluate(trainer):
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    # log and save the evaluation metrics
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

# function to perform prediction (I think this is for testing)
def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing") # if there is no dataset for testing then predict_dataset = None

    elif isinstance(predict_dataset, dict): # if there exists instances in a particular dataset
        
        for dataset_name, d in predict_dataset.items(): # SEE THE FORMAt OF THE DATASET PASSED FOR EVALUATION kind of a zip in python between the dataset name and d( ig it is the data )
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict") # when there exists no instances in the dataset
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

# WHAT ??
def analysis(trainer, model_args, data_args):
    trainer.compute_hiddens(model_args, data_args )



if __name__ == '__main__':

    args = get_args() # get the arguments for the model

    model_args, data_args, training_args, _ = args # divide the arguments into model data and training args

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    # set the verbosity for logging purposes ( higher verbosity equals higher returns )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # make a path to the checkpoints folder for getting the previous checkpoints and saving the new checkpoints
    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    # get the trainer functions based on the task names assigned in the data arguments
    if data_args.task_name.lower() == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "ner":
        assert data_args.dataset_name.lower() in NER_DATASETS
        from tasks.ner.get_trainer import get_trainer

    elif data_args.task_name.lower() == "srl":
        assert data_args.dataset_name.lower() in SRL_DATASETS
        from tasks.srl.get_trainer import get_trainer
    
    elif data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS # we work with this
        from tasks.qa.get_trainer import get_trainer

    elif data_args.task_name.lower() == "pos":
        assert data_args.dataset_name.lower() in POS_DATASETS
        from tasks.pos.get_trainer import get_trainer
        
    else:
        raise NotImplementedError('Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed) # set seed to get the results same for all the runs and experiments during compute

    trainer, predict_dataset = get_trainer(args) # get the trainer from the return of the get_trainer python file, also get the formatted dataset to predict

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    if training_args.do_train: # if the code asks us to perform the training
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint) # call the train() function on the trainer that we got from the get_trainer.py
    
    if training_args.do_eval and (training_args.resume_from_checkpoint) and (not data_args.do_analysis): # if we have to perform the evaluation then this
         from transformers.file_utils import WEIGHTS_NAME
         #if not os.path.isfile(os.path.join(training_args.resume_from_checkpoint, WEIGHTS_NAME)):
         #       raise ValueError(f"Can't find a valid checkpoint at {training_args.resume_from_checkpoint}")
         logger.info(f"Loading model from {training_args.resume_from_checkpoint}).") # load model from a given checkpoint
         """
         ### for transformer 4.13  -lifu
         state_dict = torch.load(os.path.join(training_args.resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
         # If the model is on the GPU, it still works!
         trainer.model.load_state_dict(state_dict)
         ## trainer._load_state_dict_in_model(state_dict)  # old version
         # release memory
         del state_dict
         """
         ### for transformer 4.20  -lifu
         trainer._load_from_checkpoint(training_args.resume_from_checkpoint)

         evaluate(trainer) # call the evaluate function on the trainer where the weights are loaded from the checkpoint

    if data_args.do_analysis and (training_args.resume_from_checkpoint):
         ## test for large model only -lifu
         from transformers.file_utils import WEIGHTS_NAME
         if not os.path.isfile(os.path.join(training_args.resume_from_checkpoint, WEIGHTS_NAME)):
                raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")
         logger.info(f"Loading model from {training_args.resume_from_checkpoint}).")
         state_dict = torch.load(os.path.join(training_args.resume_from_checkpoint, WEIGHTS_NAME), map_location="cpu")
         
        # IDK what is happening in this part of the code??
         keys = set(state_dict.keys())
         #keys_to_remove = ["classifier.out_proj.bias", "classifier.out_proj.weight", "classifier.dense.weight", "classifier.dense.bias"]
         for key in keys:
             if "classifier" in key:
                  print('unload', key)
                  state_dict.pop(key)
         # If the model is on the GPU, it still works!
         trainer._load_state_dict_in_model(state_dict)
         # release memory
         del state_dict
         analysis(trainer, model_args, data_args) # perform analysis if asked to analyse the model

    # if training_args.do_predict:
    #     predict(trainer, predict_dataset)

   
