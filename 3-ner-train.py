import logging
import os
from typing import List

import torch
import typer
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForTokenClassification

import nlpbook
from chrisbase.data import AppTyper, JobTimer, ArgumentsUsing, RuntimeChecking
from chrisbase.io import hr
from nlpbook.arguments import TrainerArguments
from nlpbook.ner import NERCorpus, NERDataset, NERTask

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def train(
        # env
        project: str = typer.Option(default="DeepKNLU"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        # data
        data_name: str = typer.Option(default="klue-ner-mini"),
        train_file: str = typer.Option(default="klue-ner-v1.1_train.jsonl"),
        valid_file: str = typer.Option(default="klue-ner-v1.1_dev.jsonl"),
        test_file: str = typer.Option(default=None),
        num_check: int = typer.Option(default=2),
        # model
        pretrained: str = typer.Option(default="klue/roberta-small"),
        model_name: str = typer.Option(default="{g_epoch:04.1f}, {val_loss:06.4f}, {val_F1c:05.2f}, {val_F1e:05.2f}"),
        seq_len: int = typer.Option(default=64),
        # hardware
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="32-true"),
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        batch_size: int = typer.Option(default=100),
        # learning
        validate_fmt: str = typer.Option(default="loss={val_loss:06.4f}, F1c={val_F1c:05.2f}, F1e={val_F1e:05.2f}"),
        validate_on: float = typer.Option(default=0.25),
        save_by: str = typer.Option(default="max val_F1c"),
        epochs: int = typer.Option(default=1),
        lr: float = typer.Option(default=5e-5),
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('high')
    logging.getLogger("fsspec.local").setLevel(logging.WARNING)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
    args = TrainerArguments.from_args(
        project=project,
        job_name=job_name,
        debugging=debugging,
        data_name=data_name,
        train_file=train_file,
        valid_file=valid_file,
        test_file=test_file,
        num_check=num_check,
        pretrained=pretrained,
        model_name=model_name,
        seq_len=seq_len,
        accelerator=accelerator,
        precision=precision,
        strategy=strategy,
        device=device,
        batch_size=batch_size,
        validate_fmt=validate_fmt,
        validate_on=validate_on,
        save_by=save_by,
        epochs=epochs,
        lr=lr,
    )
    with JobTimer(f"python {args.env.running_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        with ArgumentsUsing(args):
            args.info_args().set_seed()
            corpus = NERCorpus(args)
            tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
            assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
            logger.info(hr('-'))

            train_dataset = NERDataset("train", corpus=corpus, tokenizer=tokenizer)
            train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                          num_workers=args.hardware.cpu_workers,
                                          batch_size=args.hardware.batch_size,
                                          collate_fn=corpus.encoded_examples_to_batch,
                                          drop_last=False)
            logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
            logger.info(f"Created train_dataloader providing {len(train_dataloader)} batches")
            logger.info(hr('-'))

            valid_dataset = NERDataset("valid", corpus=corpus, tokenizer=tokenizer)
            valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                          num_workers=args.hardware.cpu_workers,
                                          batch_size=args.hardware.batch_size,
                                          collate_fn=corpus.encoded_examples_to_batch,
                                          drop_last=False)
            logger.info(f"Created valid_dataset providing {len(valid_dataset)} examples")
            logger.info(f"Created valid_dataloader providing {len(valid_dataloader)} batches")
            logger.info(hr('-'))

            pretrained_model_config = AutoConfig.from_pretrained(
                args.model.pretrained,
                num_labels=corpus.num_labels
            )
            model = AutoModelForTokenClassification.from_pretrained(
                args.model.pretrained,
                config=pretrained_model_config
            )
            logger.info(hr('-'))

            with RuntimeChecking(args.configure_csv_logger()):
                trainer: Trainer = nlpbook.make_trainer(args)
                trainer.fit(NERTask(args,
                                    model=model,
                                    trainer=trainer,
                                    epoch_steps=len(train_dataloader),
                                    valid_dataset=valid_dataset),
                            train_dataloaders=train_dataloader,
                            val_dataloaders=valid_dataloader)


if __name__ == "__main__":
    app()
