import logging
import os
from typing import List

import torch
import typer
from flask import Flask
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput

import nlpbook
from chrisbase.data import AppTyper, JobTimer, ArgumentsUsing, RuntimeChecking
from chrisbase.io import hr
from nlpbook.arguments import TrainerArguments, TesterArguments, ServerArguments
from nlpbook.cls import NsmcCorpus, ClassificationDataset, ClassificationTask

logger = logging.getLogger(__name__)
app = AppTyper()


@app.command()
def train(
        # env
        project: str = typer.Option(default="DeepKNLU"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        # data
        data_name: str = typer.Option(default="nsmc-mini"),
        train_file: str = typer.Option(default="ratings_train.txt"),
        valid_file: str = typer.Option(default="ratings_test.txt"),
        test_file: str = typer.Option(default=None),
        num_check: int = typer.Option(default=2),
        # model
        pretrained: str = typer.Option(default="pretrained/KPF-BERT"),
        model_name: str = typer.Option(default="{ep:3.1f}, {val_loss:06.4f}, {val_acc:06.4f}"),
        seq_len: int = typer.Option(default=64),
        # hardware
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="16-mixed"),
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        batch_size: int = typer.Option(default=64),
        # learning
        validate_fmt: str = typer.Option(default="loss={val_loss:06.4f}, acc={val_acc:06.4f}"),
        validate_on: float = typer.Option(default=0.1),
        num_save: int = typer.Option(default=3),
        save_by: str = typer.Option(default="max val_acc"),
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
        num_save=num_save,
        save_by=save_by,
        epochs=epochs,
        lr=lr,
    )
    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        with ArgumentsUsing(args):
            args.info_args().set_seed()
            corpus = NsmcCorpus(args)
            tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
            logger.info(hr('-'))

            train_dataset = ClassificationDataset("train", corpus=corpus, tokenizer=tokenizer)
            train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset, replacement=False),
                                          num_workers=args.hardware.cpu_workers,
                                          batch_size=args.hardware.batch_size,
                                          collate_fn=nlpbook.data_collator,
                                          drop_last=False)
            logger.info(f"Created train_dataset providing {len(train_dataset)} examples")
            logger.info(f"Created train_dataloader providing {len(train_dataloader)} batches")
            logger.info(hr('-'))

            valid_dataset = ClassificationDataset("valid", corpus=corpus, tokenizer=tokenizer)
            valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset),
                                          num_workers=args.hardware.cpu_workers,
                                          batch_size=args.hardware.batch_size,
                                          collate_fn=nlpbook.data_collator,
                                          drop_last=False)
            logger.info(f"Created valid_dataset providing {len(valid_dataset)} examples")
            logger.info(f"Created valid_dataloader providing {len(valid_dataloader)} batches")
            logger.info(hr('-'))

            pretrained_model_config = AutoConfig.from_pretrained(
                args.model.pretrained,
                num_labels=corpus.num_labels
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model.pretrained,
                config=pretrained_model_config,
            )
            logger.info(hr('-'))

            with RuntimeChecking(args.configure_csv_logger()):
                trainer: Trainer = nlpbook.make_trainer(args)
                trainer.fit(ClassificationTask(args,
                                               model=model,
                                               trainer=trainer,
                                               epoch_steps=len(train_dataloader)),
                            train_dataloaders=train_dataloader,
                            val_dataloaders=valid_dataloader)


@app.command()
def test(
        # env
        project: str = typer.Option(default="DeepKNLU"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        # data
        data_name: str = typer.Option(default="nsmc"),
        train_file: str = typer.Option(default=None),
        valid_file: str = typer.Option(default=None),
        test_file: str = typer.Option(default="ratings_test.txt"),
        num_check: int = typer.Option(default=2),
        # model
        pretrained: str = typer.Option(default="pretrained/KPF-BERT"),
        model_name: str = typer.Option(default=None),
        seq_len: int = typer.Option(default=64),
        # hardware
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="16-mixed"),
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        batch_size: int = typer.Option(default=64),
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('high')
    logging.getLogger("fsspec.local").setLevel(logging.WARNING)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
    args = TesterArguments.from_args(
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
    )
    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        with ArgumentsUsing(args):
            args.info_args()
            corpus = NsmcCorpus(args)
            tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
            checkpoint_path = args.env.output_home / args.model.name
            assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
            logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
            logger.info(hr('-'))

            test_dataset = ClassificationDataset("test", corpus=corpus, tokenizer=tokenizer)
            test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                         num_workers=args.hardware.cpu_workers,
                                         batch_size=args.hardware.batch_size,
                                         collate_fn=nlpbook.data_collator,
                                         drop_last=False)
            logger.info(f"Created test_dataset providing {len(test_dataset)} examples")
            logger.info(f"Created test_dataloader providing {len(test_dataloader)} batches")
            logger.info(hr('-'))

            pretrained_model_config = AutoConfig.from_pretrained(
                args.model.pretrained,
                num_labels=corpus.num_labels
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model.pretrained,
                config=pretrained_model_config,
            )
            logger.info(hr('-'))

            with RuntimeChecking(args.configure_csv_logger()):
                tester: Trainer = nlpbook.make_tester(args)
                tester.test(ClassificationTask(args,
                                               model=model,
                                               trainer=tester,
                                               epoch_steps=len(test_dataloader)),
                            dataloaders=test_dataloader,
                            ckpt_path=checkpoint_path)


@app.command()
def serve(
        # env
        project: str = typer.Option(default="DeepKNLU"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        # data
        data_name: str = typer.Option(default="nsmc"),
        train_file: str = typer.Option(default=None),
        valid_file: str = typer.Option(default=None),
        test_file: str = typer.Option(default="ratings_test.txt"),
        num_check: int = typer.Option(default=2),
        # model
        pretrained: str = typer.Option(default="pretrained/KPF-BERT"),
        model_name: str = typer.Option(default=None),
        seq_len: int = typer.Option(default=64),
        # hardware
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="16-mixed"),
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        batch_size: int = typer.Option(default=64),
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_float32_matmul_precision('high')
    logging.getLogger("fsspec.local").setLevel(logging.WARNING)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARNING)
    args = ServerArguments.from_args(
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
    )
    with JobTimer(f"python {args.env.current_file} {' '.join(args.env.command_args)}", rt=1, rb=1, rc='=', verbose=True, flush_sec=0.3):
        with ArgumentsUsing(args):
            args.info_args()
            corpus = NsmcCorpus(args)
            tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
            checkpoint_path = args.env.output_home / args.model.name
            assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
            logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
            checkpoint: dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            logger.info(hr('-'))

            pretrained_model_config = AutoConfig.from_pretrained(
                args.model.pretrained,
                num_labels=corpus.num_labels
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model.pretrained,
                config=pretrained_model_config,
            )
            model.load_state_dict({k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()})
            model.eval()
            logger.info(hr('-'))

            def inference_fn(sentence):
                inputs = tokenizer(
                    [sentence],
                    max_length=args.model.seq_len,
                    padding="max_length",
                    truncation=True,
                )
                with torch.no_grad():
                    outputs: SequenceClassifierOutput = model(**{k: torch.tensor(v) for k, v in inputs.items()})
                    prob = outputs.logits.softmax(dim=1)
                    positive_prob = round(prob[0][1].item(), 4)
                    negative_prob = round(prob[0][0].item(), 4)
                    pred = "긍정 (positive)" if torch.argmax(prob) == 1 else "부정 (negative)"
                return {
                    'sentence': sentence,
                    'prediction': pred,
                    'positive_data': f"긍정 {positive_prob:.4f}",
                    'negative_data': f"부정 {negative_prob:.4f}",
                    'positive_width': f"{positive_prob * 100}%",
                    'negative_width': f"{negative_prob * 100}%",
                }

            with RuntimeChecking(args.configure_csv_logger()):
                server: Flask = nlpbook.make_server(inference_fn,
                                                    template_file="serve_cls.html",
                                                    ngrok_home=args.env.working_path)
                server.run()


if __name__ == "__main__":
    app()
