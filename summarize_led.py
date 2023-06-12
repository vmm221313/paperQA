import os
import re
import glob
import torch
from tqdm import tqdm

from transformers import LEDForConditionalGeneration, LEDTokenizer


def main():
    papers = []
    names = []
    for path in glob.glob("data/papers_txt/*.txt"):
        names.append(path.split("/")[-1][:-4])
        with open(path, "r") as f:
            papers.append(re.sub("\n", " ", " ".join(f.readlines()).strip()))


    tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed", return_dict_in_generate=True).to("cuda").half()

    # summarize papers taking only the first 16384 tokens
    # (we apply this limit because the context length in Longformer
    # is limited to 16384 tokens)
    summaries = []
    for text in papers:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        input_ids = input_ids[:, :16384].to("cuda")
        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask[:, 0] = 1

        sequences = model.generate(input_ids, global_attention_mask=global_attention_mask, max_length=1000).sequences
        summaries.append(tokenizer.batch_decode(sequences))

    path = "data/summaries_led_first_16384_txt/"
    os.makedirs(path, exist_ok=True)
    for idx, summary in enumerate(summaries):
        with open(f"{path}/{names[idx]}.txt", "w") as f:
            f.write(summary[0])


    # alternatively, we can split up the paper into chunks 
    # of 1000 tokens and summarize each chunk separately 
    tokenizer = LEDTokenizer.from_pretrained("patrickvonplaten/led-large-16384-pubmed")
    model = LEDForConditionalGeneration.from_pretrained("patrickvonplaten/led-large-16384-pubmed", return_dict_in_generate=True).to("cuda").half()

    summaries = []
    names_1000 = []
    for jdx, text in enumerate(papers):
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        for idx in tqdm(range(0, input_ids.shape[1], 1000)):
            current_input_ids = input_ids[:, idx : idx+1000].to("cuda")

            global_attention_mask = torch.zeros_like(current_input_ids)
            global_attention_mask[:, 0] = 1

            sequences = model.generate(current_input_ids, global_attention_mask=global_attention_mask, max_new_tokens=1000).sequences
            summaries.append(tokenizer.batch_decode(sequences))

            names_1000.append(names[jdx])

    path = "data/summaries_1000_chunks_txt/"
    os.makedirs(path, exist_ok=True)
    for idx, summary in enumerate(summaries): 
        with open(f"{path}/{names_1000[idx]}_{idx}.txt", "w") as f:
            f.write(summary[0])


if __name__ == "__main__":
    main()