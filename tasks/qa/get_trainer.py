# import required modules
import logging
import os
import random
import sys

# from the transformers huggingface import Autoconfig(change the config of the model) and AutoTokenizer(tokenizer for the model being used)
from transformers import (
    AutoConfig,
    AutoTokenizer,
)


# import the models for multilingual T5 and the tokenizer corresponding to it
from transformers import MT5ForConditionalGeneration, T5Tokenizer, PreTrainedTokenizerFast, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

# from the qa.dataset.py import squad and squad seq2seq 
from tasks.qa.dataset import SQuAD, SQuAD_seq2seq

# get the trainer functions that are based on the trainers from the huggingface model library
from training.trainer_qa import QuestionAnsweringTrainer
from training.trainer_seq2seq_qa import QuestionAnsweringSeq2seqTrainer

# get the model and the task type from utils.py
from model.utils import get_model, TaskType

logger = logging.getLogger(__name__)

# the get trainer function to write the trainer that would be called from run.py
def get_trainer(args):
    model_args, data_args, training_args, qa_args = args # get the arguments

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, # check if there is some preferred model or is there a path to the model
        num_labels=2, # IDK labels for what ??
        revision=model_args.model_revision, # IDK what this arg stands for ?? (helps in revising the model version or something like version control system)
    )

    # corrected offset mappings
    from transformers import XLMRobertaTokenizerFast # tokenizer for the base model XLM roberta when no model is predefined
    #tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", from_slow=True)

    if 'xl' in model_args.model_name_or_path: # if there is some predefined model for the task then get the tokenizer for the particular task
        tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
        )
    else:
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base", from_slow=True) # else get the tokenizer for the XLM-Roberta
    # 

    """
    ## this could give you wrong offset mapping ( exclude space for a word with space) -Lifu
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        use_fast=True,
    )
    """
    if model_args.prefix or model_args.prompt: # if prefix tuning is turned on then do the following...
        model = get_model(model_args, TaskType.QUESTION_ANSWERING, config, fix_bert=True) # get model called from utils.py

        dataset = SQuAD(tokenizer, data_args, training_args, qa_args) # get the squad dataset and format in the required manner

    else:
        if 'mt5' in model_args.model_name_or_path:
            # add later -Lifu
            print('mt5')
            training_args.generation_max_length = 30
            training_args.predict_with_generate = True
            #training_args.generation_num_beams = 5
            #model = MT5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
            #tokenizer = PreTrainedTokenizerFast.from_pretrained(model_args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
            dataset = SQuAD_seq2seq(tokenizer, data_args, training_args, qa_args)
           
        else:
            model = get_model(model_args, TaskType.QUESTION_ANSWERING, config, fix_bert=False)

            dataset = SQuAD(tokenizer, data_args, training_args, qa_args)

    if 'mt5' in model_args.model_name_or_path: # if there is MT5 in the model name or path then ... (IDK why this is done though ??)

        data_collator = DataCollatorForSeq2Seq(
          tokenizer,
          model=model,
          label_pad_token_id=-100,
          pad_to_multiple_of=8 if training_args.fp16 else None,
        ) # design a data collator - form a batch for the training purposes using the elements of the dataset as present in the train and the validation set (imported directly from transformers)
        
        trainer = QuestionAnsweringSeq2seqTrainer(
          model=model,
          args=training_args,
          train_dataset=dataset.train_dataset if training_args.do_train else None,
          eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
          eval_examples=dataset.eval_examples if training_args.do_eval else None,
          tokenizer=tokenizer,
          data_collator=data_collator,
          post_process_function=dataset.post_processing_function,
          compute_metrics=dataset.compute_metrics,
        ) # from the seq2seq trainer for the question answering task (see the trainer function for more reference)

    else:

        trainer = QuestionAnsweringTrainer(
          model=model,
          args=training_args,
          train_dataset=dataset.train_dataset if training_args.do_train else None,
          eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
          eval_examples=dataset.eval_examples if training_args.do_eval else None,
          tokenizer=tokenizer,
          data_collator=dataset.data_collator,
          post_process_function=dataset.post_processing_function,
          compute_metrics=dataset.compute_metrics,
        )

    return trainer, dataset.predict_dataset


