# Problem Statement

The goal of this project is to answer questions from [medical PDFs](https://www.cdc.gov/vaccines/hcp/acip-recs/vacc-specific/hpv.html). We restrict our analysis to open source models. The questions are: 

- When did the GARDASIL 9 recommendations change?
- What were the past 3 recommendation changes for GARDASIL 9?
- Is GARDASIL 9 recommended for Adults?
- Does the ACIP recommend one dose GARDASIL 9?


# Methodology 

## Data Preparation 

We use the [PyPDF2](https://pypdf2.readthedocs.io/en/3.0.0/) library to extract text from the given PDFs via OCR. The following snippet shows the basic approach, and the complete code can by found in parse_pdfs.py

```
from PyPDF2 import PdfReader

reader = PdfReader(paper)

text = []
for page_idx in range(len(reader.pages)):
    text.append(reader.pages[page_idx].extract_text())

text = " ".join(text)
text = re.sub("\n", "", text)       
```


## Approach

We use the [Retrieval Augmented Generation](https://arxiv.org/pdf/2005.11401.pdf) paradigm for this task. Given a question, a fixed number of relevant contexts are retrieved from the PDFs using the [Dense Passage Retriever](https://arxiv.org/pdf/2004.04906.pdf). These are then concatenated with the question and fed to a generative language model to generate a free-form (aka abstractive) answer for the given question via the text-to-text generation paradigm. 


# Experiments

## Baseline

For the baseline, we use apply RAG model with a [DPR](https://arxiv.org/pdf/2004.04906.pdf) retriever and a [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) text-to-text generative model. The following code snippet outlines the basic approach. For the complete pipeline refer to rag_pipeline.py. 
```
# initialize context encoder
context_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(args.model.context_encoder_config)
context_encoder = DPRContextEncoder.from_pretrained(args.model.context_encoder_config).to(device=device)

# initialize question encoder
question_encoder = DPRQuestionEncoder.from_pretrained(args.model.question_encoder_config) 
question_encoder_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(args.model.question_encoder_config)

# compute the context embeddings
new_features = Features({"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))})
dataset = dataset.map(
    partial(embed, context_encoder=context_encoder, context_tokenizer=context_tokenizer),
    batched=True,
    batch_size=1,
    features=new_features,
)
index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
index_HF = CustomHFIndex(vector_size=768, dataset=dataset)
dataset.add_faiss_index("embeddings", custom_index=index)

# initialize generator
generator_tokenizer = AutoTokenizer.from_pretrained(args.model.generator_config)
generator = AutoModelForSeq2SeqLM.from_pretrained(args.model.generator_config).to(device).half()

# initialize retriever
retriever = RagRetriever(
                        config=RagConfig.from_pretrained("facebook/rag-token-nq"), 
                        question_encoder_tokenizer=question_encoder_tokenizer,
                        generator_tokenizer=generator_tokenizer,
                        index=index_HF)

# combine question_encoder, retriever, and generator into one single model
model = RagSequenceForGeneration(question_encoder=question_encoder, retriever=retriever, generator=generator).to(device)

# encode question
print(f"question: {question}")
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
print(f"answer: {generated_string}") 
```


|      	| contexts       	| join  	| split_every 	| retriever 	| num_retrieved 	| generator     	| When did the GARDASIL 9 recommendations change? 	| What were the past 3 recommendation changes for GARDASIL 9? 	| Is GARDASIL 9 recommended for Adults? 	| Does the ACIP recommend one dose GARDASIL 9? 	|
|------	|----------------	|-------	|-------------	|-----------	|---------------	|---------------	|-------------------------------------------------	|-------------------------------------------------------------	|---------------------------------------	|----------------------------------------------	|
| v1.0 	| raw paper text 	| FALSE 	| 32          	| DPR       	| 10            	| Flan-T5 Large 	| February 2015                                   	| Recommendations were approved by ACIP in February 2015.     	| no                                    	| yes                                          	|
|      	|                	|       	|             	|           	|               	|               	|                                                 	|                                                             	|                                       	|                                              	|
|      	|                	|       	|             	|           	|               	|               	|                                                 	|                                                             	|                                       	|                                              	|


It can be observed that the performance is reasonably good even without any fine-tuning or augmentation, with reasonable answers to three out of the four questions. 


## Concat + Chunks

In this experiment, we apply two modifications: 1) concatenate all the contexts together, and 2) split the combined context into chunks of a fixed number of tokens. This simple trick has been shown to improve retrieval performance by allowing DPR to select smaller, more fine-grained contexts from the entire database. 

|      	| contexts       	| join  	| split_every 	| retriever 	| num_retrieved 	| generator     	| When did the GARDASIL 9 recommendations change? 	| What were the past 3 recommendation changes for GARDASIL 9? 	| Is GARDASIL 9 recommended for Adults?         	| Does the ACIP recommend one dose GARDASIL 9?             	|
|------	|----------------	|-------	|-------------	|-----------	|---------------	|---------------	|-------------------------------------------------	|-------------------------------------------------------------	|-----------------------------------------------	|----------------------------------------------------------	|
| v1.0 	| raw paper text 	| FALSE 	| 32          	| DPR       	| 10            	| Flan-T5 Large 	| February 2015                                   	| Recommendations were approved by ACIP in February 2015.     	| no                                            	| yes                                                      	|
| v1.4 	| raw paper text 	| FALSE 	| 128         	| DPR       	| 10            	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| 9vHPV, 4vHPV or 2vHPV can be used for routine 	| yes                                                      	|
| v1.5 	| raw paper text 	| FALSE 	| 64          	| DPR       	| 10            	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| yes                                           	| yes                                                      	|
|      	|                	|       	|             	|           	|               	|               	|                                                 	|                                                             	|                                               	|                                                          	|
| v1.1 	| raw paper text 	| TRUE  	| 32          	| DPR       	| 10            	| Flan-T5 Large 	| February 2015                                   	| Use of a 2-Dose Schedule for Human Papillomavirus           	| yes                                           	| yes                                                      	|
| v1.2 	| raw paper text 	| TRUE  	| 64          	| DPR       	| 10            	| Flan-T5 Large 	| 2011                                            	| Use of a 2-Dose Schedule for Human Papillomavirus           	| yes                                           	| no                                                       	|
| v1.3 	| raw paper text 	| TRUE  	| 128         	| DPR       	| 10            	| Flan-T5 Large 	| 2-dose series.                                  	| HIV                                                         	| CIN2, vulvar intraepithelial neoplasia grade  	| Human Papillomavirus Vaccination Recommendations of the  	|



## Increasing the number of retrieved contexts  
In this experiment we try increasing the number of contexts retrieved by the DPR retriever. Interestingly, it can be observed that there's no performance improvement on increasing the number of retrieved contexts. We verified that this wasn't due to a bug by checking the retrieved contexts. It turns out that this is because the answer is mostly in the first retrieved context (i.e. most similar context) itself. This indicates that the DPR retriever is able to retrieve very relevant contexts most of the time. 

|       	| contexts       	| join  	| split_every 	| retriever 	| num_retrieved 	| generator     	| When did the GARDASIL 9 recommendations change? 	| What were the past 3 recommendation changes for GARDASIL 9? 	| Is GARDASIL 9 recommended for Adults?         	| Does the ACIP recommend one dose GARDASIL 9? 	|
|-------	|----------------	|-------	|-------------	|-----------	|---------------	|---------------	|-------------------------------------------------	|-------------------------------------------------------------	|-----------------------------------------------	|----------------------------------------------	|
| v1.8  	| raw paper text 	| FALSE 	| 64          	| DPR       	| 1             	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| 9vHPV, 4vHPV or 2vHPV can be used for routine 	| yes                                          	|
| v1.9  	| raw paper text 	| FALSE 	| 64          	| DPR       	| 5             	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| yes                                           	| yes                                          	|
| v1.10 	| raw paper text 	| FALSE 	| 64          	| DPR       	| 10            	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| yes                                           	| yes                                          	|
| v1.11 	| raw paper text 	| FALSE 	| 64          	| DPR       	| 20            	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| yes                                           	| yes                                          	|


## Summarizing the contexts 
From the parsed papers, it can be observed that the text is very noisy. This might make it difficult for the DPR retriever to identify the relevant sections. Hence, we try to clean up the contexts by summarizing them using finetuned models like [Longformer-Encoder-Decoder](https://arxiv.org/abs/2004.05150) and [Big Bird](https://arxiv.org/abs/2007.14062), which were designed to handle long inputs. It can be observed that the performance for the second question improves slightly, with atleast one reasonable reason being generated by the model. 


|       	| contexts                                	| join  	| split_every 	| retriever 	| num_retrieved 	| generator     	| When did the GARDASIL 9 recommendations change? 	| What were the past 3 recommendation changes for GARDASIL 9?                  	| Is GARDASIL 9 recommended for Adults?        	| Does the ACIP recommend one dose GARDASIL 9?                                 	|
|-------	|-----------------------------------------	|-------	|-------------	|-----------	|---------------	|---------------	|-------------------------------------------------	|------------------------------------------------------------------------------	|----------------------------------------------	|------------------------------------------------------------------------------	|
| v1.0  	| raw paper text                          	| FALSE 	| 32          	| DPR       	| 10            	| Flan-T5 Large 	| February 2015                                   	| Recommendations were approved by ACIP in February 2015.                      	| no                                           	| yes                                                                          	|
| v1.15 	| summaries with LED (first 16384 tokens) 	| FALSE 	| 32          	| DPR       	| 10            	| Flan-T5 Large 	| 2011                                            	| Use of a 2-Dose Schedule for Human Papillomavirus                            	| Advisory Committee on Immunization Practices 	| Advisory Committee on Immunization Practices                                 	|
| v1.14 	| summaries with LED (first 16384 tokens) 	| FALSE 	| 64          	| DPR       	| 10            	| Flan-T5 Large 	| 2011                                            	| Use of a 2-Dose Schedule for Human Papillomavirus                            	| CDC                                          	| Advisory Committee on Immunization Practices                                 	|
| v1.17 	| summaries Pegasus (1024 chunks)         	| FALSE 	| 64          	| DPR       	| 10            	| Flan-T5 Large 	| December 10, 2014                               	| Use of a 2-Dose Schedule for Human Papillomavirus                            	| no                                           	| FDA approved 9vHPV for use in a 2-dose series for girls and boys aged        	|
| v1.18 	| summaries (BigBird) (4096 chunks)       	| FALSE 	| 64          	| DPR       	| 10            	| Flan-T5 Large 	| s>                                              	| cdc recommends universal vaccination for adults aged 16 years, whereas the c 	| s>                                           	| cdc recommends universal vaccination for adults aged 16 years, whereas the c 	|


# More ideas

- Finetuning the generative model on the given dataset in an self-supervised manner using masked language modelling
- Finetuning the model on a related dataset like [QASPER](https://allenai.org/data/qasper) to improve question answering performance (might be difficult for large models like Flan-T5 which we use in our experiments)
- Trying the summarization experiment discussed above but with larger open-source LLMs like [Falcon-7B](https://huggingface.co/tiiuae)

# Reproducing the results

1. Install the required libraries. Note: we use the FAISS library which needs to be installed as explained [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
```
pip install -r requirements.txt
```

2. For parsing the PDFs, run:
```
python parse_pdfs.py 
```

3. To summarize the papers using [Longformer-Encoder-Decoder](https://arxiv.org/abs/2004.05150) run:
```
python summarize_led.py 
```

4. To summarize the papers using [Big Bird](https://arxiv.org/abs/2007.14062) run:
```
python summarize_big_bird.py 
```

5. Finally, run the RAG pipeline by specifying the parameters in a config file (.yaml) as follows: 
```
python rag_pipeline.py --config configs/base.yaml
```

Each parameter is explained below:
```
data:
  questions_file: data/questions.txt                                        # .txt file with questions 
  contexts_dir: data/summaries_pegasus_pubmed_4096_chunks_txt/              # directory with contexts (full papers or summaries) as separate .txt files

  join_contexts: False                                                      # flag indicating whether the contexts should be joined together to form a single long context
  split: True                                                               # flag indicating whether the contexts should be split into chunks
  split_every: 32                                                           # number of tokens in each chunk

model:
  # question_encoder  
  question_encoder_config: "facebook/dpr-question_encoder-single-nq-base"   # huggingface pretrained model to use for encoding the question

  # retriever
  num_retrieved: 10                                                         # number of contexts to retrieve for each question
  context_encoder_config: "facebook/dpr-ctx_encoder-single-nq-base"         # huggingface pretrained model to use for encoding the context
  
  # generator
  generator_config: "google/flan-t5-large"                                  # huggingface pretrained model to use for generating the answer

artifact_path: exps/v1.18/                                                  # directory to save experiment artifacts
```