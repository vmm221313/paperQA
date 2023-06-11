import os
import re
import json
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from typing import List, Optional
from omegaconf import OmegaConf

import torch
import torch.nn as nn 

import faiss

from datasets import (
    Dataset, 
    Features,
    Sequence,
    Value, 
    load_dataset
)
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoderTokenizerFast,
    RagRetriever,
    RagTokenForGeneration,
    RagSequenceForGeneration,
    RagTokenizer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_questions(path):
    questions = []
    with open(f"{path}", "r") as f:
        questions = f.readlines()

    questions = [q.strip() for q in questions]
    return questions


def load_contexts(path):
    titles = []
    contexts = []
    for path in glob.glob(f"{path}/*.txt"):
        titles.append(path.split("/")[-1][:-4])
        with open(f"{path}", "r") as f:
            contexts.append(" ".join(f.readlines()))

    return titles, contexts


def save_answers(path, answers):
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/answers.txt", "w") as f:
        for ans in answers:
            f.write(ans.strip() + "\n")


def split_text(text: str, n, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents: dict, n: int) -> dict:
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text, n):
                titles.append(title if title is not None else "")
                texts.append(passage)
    
    return {"title": titles, "text": texts}


def join_contexts(titles, texts):
    titles = [" ".join(titles)]
    texts = [" ".join(texts)]

    return titles, texts


def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt", is_split_into_words=True, max_length=128)["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output

    return {"embeddings": embeddings.detach().cpu().numpy()}


def main(args):
    print("args:", args, "\n")

    questions = load_questions(args.data.questions_file)
    print("questions:", "\n", questions, "\n")

    titles, contexts = load_contexts(args.data.contexts_dir)
    print("len(titles):", len(titles), "len(contexts):", len(contexts), "\n")

    if (args.data.join_contexts):
        titles, contexts = join_contexts(titles, contexts)

    dataset = Dataset.from_dict({"title": titles, "text": contexts})
    print(dataset)

    if (args.data.split):
        dataset = dataset.map(
                            partial(split_documents, n=args.data.split_every),
                            batched=True,
                            batch_size=1,
        )
        print(dataset)

    ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device=device)
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

    # compute the embeddings
    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})
    dataset = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=1,
        features=new_features,
    )
    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)
    print(dataset)
    print(dataset.get_index("embeddings"))

    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", indexed_dataset=dataset)
    model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever).to(device)
    
    answers = []
    for question in questions:
        print(f"question: {question}")
        generated = model.generate(tokenizer.question_encoder(question, return_tensors="pt")["input_ids"].to(device))
        generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        answers.append(generated_string)
        print(f"answer: {generated_string}") 
        print()
    
    if (args.answers_dir is not None):
        save_answers(args.answers_dir, answers)
        OmegaConf.save(args, f"{args.answers_dir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="config file")
    args = parser.parse_args()
    args = OmegaConf.load(args.config)

    main(args)