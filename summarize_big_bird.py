import os
import re
import glob
from tqdm import tqdm

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM


def main():
    papers = []
    names = []
    for path in glob.glob("data/papers_txt/*.txt"):
        names.append(path.split("/")[-1][:-4])
        with open(path, "r") as f:
            papers.append(re.sub("\n", " ", " ".join(f.readlines()).strip()))

    # pegasus large: split up the paper into chunks 
    # of 1000 tokens and summarize each chunk separately  
    tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-large").cuda()

    summaries = []
    names_1024 = []
    for jdx, text in enumerate(papers):
        inputs = tokenizer(text, return_tensors="pt")
        for idx in tqdm(range(0, inputs["input_ids"].shape[1], 1024)):
            current_input_ids = inputs["input_ids"][:, idx : idx+1024].to("cuda")
            current_attention_mask = inputs["attention_mask"][:, idx : idx+1024].to("cuda")
            outputs = model.generate(input_ids=current_input_ids, attention_mask=current_attention_mask)
            summaries.append(tokenizer.batch_decode(outputs))
            names_1024.append(names[jdx])

    path = "data/summaries_pegasus_1024_chunks_txt/"
    os.makedirs(path, exist_ok=True)
    for idx, summary in enumerate(summaries): 
        with open(f"{path}/{names_1024[idx]}_{idx}.txt", "w") as f:
            f.write(summary[0])

    
    # bigbird-pegasus-large-pubmed: split up the paper into chunks 
    # of 4096 tokens and summarize each chunk separately  
    tokenizer = AutoTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
    model = BigBirdPegasusForConditionalGeneration.from_pretrained("google/bigbird-pegasus-large-pubmed").cuda()

    summaries = []
    names_4096 = []
    for jdx, text in enumerate(papers):
        inputs = tokenizer(text, return_tensors="pt")
        for idx in tqdm(range(0, inputs["input_ids"].shape[1], 4096)):
            current_input_ids = inputs["input_ids"][:, idx : idx+4096].to("cuda")
            current_attention_mask = inputs["attention_mask"][:, idx : idx+4096].to("cuda")
            outputs = model.generate(input_ids=current_input_ids, attention_mask=current_attention_mask)
            summaries.append(tokenizer.batch_decode(outputs))
            names_4096.append(names[jdx])

    path = "data/summaries_pegasus_pubmed_4096_chunks_txt/"
    os.makedirs(path, exist_ok=True)
    for idx, summary in enumerate(summaries): 
        with open(f"{path}/{names_4096[idx]}_{idx}.txt", "w") as f:
            f.write(summary[0])


if __name__ == "__main__":
    main()