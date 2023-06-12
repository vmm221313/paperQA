import os
import re
import sys
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
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    RagConfig,
    RagRetriever,
    RagTokenForGeneration,
    RagSequenceForGeneration,
    RagTokenizer,
    AutoTokenizer, 
    AutoModelForSeq2SeqLM
)

from index import CustomHFIndex

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


def embed(documents: dict, context_encoder: DPRContextEncoder, context_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = context_tokenizer(documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt", is_split_into_words=True, max_length=128)["input_ids"]
    embeddings = context_encoder(input_ids.to(device=device), return_dict=True).pooler_output

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

    context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(args.model.context_encoder_config)
    context_encoder = DPRContextEncoder.from_pretrained(args.model.context_encoder_config).to(device=device)

    # compute the embeddings
    new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})
    dataset = dataset.map(
        partial(embed, context_encoder=context_encoder, context_tokenizer=context_tokenizer),
        batched=True,
        batch_size=1,
        features=new_features,
    )
    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    dataset.add_faiss_index("embeddings", custom_index=index)
    print(dataset)
    print(dataset.get_index("embeddings"))

    generator_tokenizer = AutoTokenizer.from_pretrained(args.model.generator_config)
    generator = AutoModelForSeq2SeqLM.from_pretrained(args.model.generator_config).to(device).half()
    
    question_encoder = DPRQuestionEncoder.from_pretrained(args.model.question_encoder_config) 
    question_encoder_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(args.model.question_encoder_config)
    
    index_HF = CustomHFIndex(vector_size=768, dataset=dataset)
    retriever = RagRetriever(
                            config=RagConfig.from_pretrained("facebook/rag-token-nq"), 
                            question_encoder_tokenizer=question_encoder_tokenizer,
                            generator_tokenizer=generator_tokenizer,
                            index=index_HF)
    
    model = RagSequenceForGeneration(question_encoder=question_encoder, retriever=retriever, generator=generator).to(device)
    
    answers = []
    for question in questions:
        print(f"question: {question}")

        # encode question
        question_tokenized = question_encoder_tokenizer(question, return_tensors="pt")
        question_input_ids = question_tokenized["input_ids"].to(device)
        question_attention_mask = question_tokenized["attention_mask"].to(device)
        question_hidden_states = question_encoder(input_ids=question_input_ids, attention_mask=question_attention_mask).pooler_output

        # retrieve relevant contexts
        contexts_dict = retriever(
                                question_input_ids=question_input_ids.detach().cpu().numpy(), 
                                question_hidden_states=question_hidden_states.detach().cpu().numpy(), 
                                n_docs=args.model.num_retrieved,
                                return_tensors="pt")

        # generate free-form answer
        generated = model.generate(input_ids=question_input_ids, attention_mask=question_attention_mask, context_input_ids=contexts_dict["context_input_ids"].to(device), context_attention_mask=contexts_dict["context_attention_mask"].to(device))
        generated_string = generator_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

        answers.append(generated_string)
        print(f"answer: {generated_string}") 
        print()
    
    if (args.artifact_path is not None):
        save_answers(args.artifact_path, answers)
        OmegaConf.save(args, f"{args.artifact_path}/config.yaml")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml", help="config file")
    args = parser.parse_args()
    args = OmegaConf.load(args.config)

    main(args)