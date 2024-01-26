import logging
import os
from pathlib import Path
from typing import List

import torch
import typer
from flask import Flask
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import PreTrainedTokenizerFast, AutoTokenizer, AutoConfig, AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput

import nlpbook
from chrisbase.data import AppTyper, JobTimer, ArgumentsUsing, RuntimeChecking
from chrisbase.io import hr
from nlpbook.arguments import TrainerArguments, TesterArguments, ServerArguments
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
        data_name: str = typer.Option(default="klue-ner-mini"),  # "kmou-ner-mini"
        train_file: str = typer.Option(default="klue-ner-v1.1_train.jsonl"),  # "train.jsonl"
        valid_file: str = typer.Option(default="klue-ner-v1.1_dev.jsonl"),  # "valid.jsonl"
        test_file: str = typer.Option(default=None),
        num_check: int = typer.Option(default=2),
        # model
        pretrained: str = typer.Option(default="klue/roberta-small"),
        model_name: str = typer.Option(default="{ep:3.1f}, {val_loss:06.4f}, {val_F1c:05.2f}, {val_F1e:05.2f}"),
        seq_len: int = typer.Option(default=64),
        # hardware
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="32-true"),
        strategy: str = typer.Option(default="auto"),
        device: List[int] = typer.Option(default=[0]),
        batch_size: int = typer.Option(default=64),
        # learning
        validate_fmt: str = typer.Option(default="loss={val_loss:06.4f}, F1c={val_F1c:05.2f}, F1e={val_F1e:05.2f}"),
        validate_on: float = typer.Option(default=0.1),
        num_save: int = typer.Option(default=3),
        save_by: str = typer.Option(default="max val_F1e"),
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
                                    test_dataset=valid_dataset),
                            train_dataloaders=train_dataloader,
                            val_dataloaders=valid_dataloader)


@app.command()
def test(
        # env
        project: str = typer.Option(default="DeepKNLU"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        # data
        data_name: str = typer.Option(default="klue-ner"),  # "kmou-ner"
        train_file: str = typer.Option(default=None),  # "train.jsonl"
        valid_file: str = typer.Option(default=None),  # "valid.jsonl"
        test_file: str = typer.Option(default="klue-ner-v1.1_dev.jsonl"),  # "valid.jsonl"
        num_check: int = typer.Option(default=2),
        # model
        pretrained: str = typer.Option(default="pretrained/KPF-BERT"),
        model_name: str = typer.Option(default="train-KPF-BERT-0906.035511"),
        seq_len: int = typer.Option(default=64),
        # hardware
        accelerator: str = typer.Option(default="gpu"),
        precision: str = typer.Option(default="32-true"),
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
            corpus = NERCorpus(args)
            tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
            assert isinstance(tokenizer, PreTrainedTokenizerFast), f"Our code support only PreTrainedTokenizerFast, but used {type(tokenizer)}"
            checkpoint_path = args.env.output_home / args.model.name
            assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
            logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
            logger.info(hr('-'))

            test_dataset = NERDataset("test", corpus=corpus, tokenizer=tokenizer)
            test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),
                                         num_workers=args.hardware.cpu_workers,
                                         batch_size=args.hardware.batch_size,
                                         collate_fn=corpus.encoded_examples_to_batch,
                                         drop_last=False)
            logger.info(f"Created test_dataset providing {len(test_dataset)} examples")
            logger.info(f"Created test_dataloader providing {len(test_dataloader)} batches")
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
                tester: Trainer = nlpbook.make_tester(args)
                tester.test(NERTask(args,
                                    model=model,
                                    trainer=tester,
                                    epoch_steps=len(test_dataloader),
                                    test_dataset=test_dataset),
                            dataloaders=test_dataloader,
                            ckpt_path=checkpoint_path)


@app.command()
def serve(
        # env
        project: str = typer.Option(default="DeepKNLU"),
        job_name: str = typer.Option(default=None),
        debugging: bool = typer.Option(default=False),
        # data
        data_name: str = typer.Option(default="klue-ner"),  # "kmou-ner"
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
            corpus = NERCorpus(args)
            tokenizer = AutoTokenizer.from_pretrained(args.model.pretrained, use_fast=True)
            checkpoint_path = args.env.output_home / args.model.name
            assert checkpoint_path.exists(), f"No checkpoint file: {checkpoint_path}"
            logger.info(f"Using finetuned checkpoint file at {checkpoint_path}")
            checkpoint: dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            label_map_path: Path = args.env.output_home / "label_map.txt"
            assert label_map_path.exists(), f"No downstream label file: {label_map_path}"
            logger.info(f"Using label map file at {label_map_path}")
            labels = label_map_path.read_text().splitlines(keepends=False)
            id_to_label = {idx: label for idx, label in enumerate(labels)}
            logger.info(hr('-'))

            pretrained_model_config = AutoConfig.from_pretrained(
                args.model.pretrained,
                num_labels=corpus.num_labels
            )
            model = AutoModelForTokenClassification.from_pretrained(
                args.model.pretrained,
                config=pretrained_model_config
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
                    outputs: TokenClassifierOutput = model(**{k: torch.tensor(v) for k, v in inputs.items()})
                    all_probs: Tensor = outputs.logits[0].softmax(dim=1)
                    top_probs, top_preds = torch.topk(all_probs, dim=1, k=1)
                    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                    top_labels = [id_to_label[pred[0].item()] for pred in top_preds]
                    result = []
                    for token, label, top_prob in zip(tokens, top_labels, top_probs):
                        if token in tokenizer.all_special_tokens:
                            continue
                        result.append({
                            "token": token,
                            "label": label,
                            "prob": f"{round(top_prob[0].item(), 4):.4f}",
                        })
                return {
                    'sentence': sentence,
                    'result': result,
                }

            with RuntimeChecking(args.configure_csv_logger()):
                server: Flask = nlpbook.make_server(inference_fn,
                                                    template_file="serve_ner.html",
                                                    ngrok_home=args.env.working_path)
                server.run()


if __name__ == "__main__":
    app()
