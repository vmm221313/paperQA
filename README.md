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

|      	| contexts       	| join  	| split_every 	| retriever 	| num_retrieved 	| generator     	| When did the GARDASIL 9 recommendations change? 	| What were the past 3 recommendation changes for GARDASIL 9? 	| Is GARDASIL 9 recommended for Adults? 	| Does the ACIP recommend one dose GARDASIL 9? 	|
|------	|----------------	|-------	|-------------	|-----------	|---------------	|---------------	|-------------------------------------------------	|-------------------------------------------------------------	|---------------------------------------	|----------------------------------------------	|
| v1.0 	| raw paper text 	| FALSE 	| 32          	| DPR       	| 10            	| Flan-T5 Large 	| February 2015                                   	| Recommendations were approved by ACIP in February 2015.     	| no                                    	| yes                                          	|
|      	|                	|       	|             	|           	|               	|               	|                                                 	|                                                             	|                                       	|                                              	|
|      	|                	|       	|             	|           	|               	|               	|                                                 	|                                                             	|                                       	|                                              	|



## Concat + Chunks



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



|       	| contexts       	| join  	| split_every 	| retriever 	| num_retrieved 	| generator     	| When did the GARDASIL 9 recommendations change? 	| What were the past 3 recommendation changes for GARDASIL 9? 	| Is GARDASIL 9 recommended for Adults?         	| Does the ACIP recommend one dose GARDASIL 9? 	|
|-------	|----------------	|-------	|-------------	|-----------	|---------------	|---------------	|-------------------------------------------------	|-------------------------------------------------------------	|-----------------------------------------------	|----------------------------------------------	|
| v1.12 	| raw paper text 	| FALSE 	| 32          	| DPR       	| 1             	| Flan-T5 Large 	| February 2015                                   	| Recommendations were approved by ACIP in February 2015.     	| no                                            	| yes                                          	|
| v1.6  	| raw paper text 	| FALSE 	| 32          	| DPR       	| 5             	| Flan-T5 Large 	| February 2015                                   	| Recommendations were approved by ACIP in February 2015.     	| no                                            	| yes                                          	|
| v1.0  	| raw paper text 	| FALSE 	| 32          	| DPR       	| 10            	| Flan-T5 Large 	| February 2015                                   	| Recommendations were approved by ACIP in February 2015.     	| no                                            	| yes                                          	|
| v1.7  	| raw paper text 	| FALSE 	| 32          	| DPR       	| 20            	| Flan-T5 Large 	| February 2015                                   	| Recommendations were approved by ACIP in February 2015.     	| no                                            	| yes                                          	|
|       	|                	|       	|             	|           	|               	|               	|                                                 	|                                                             	|                                               	|                                              	|
| v1.8  	| raw paper text 	| FALSE 	| 64          	| DPR       	| 1             	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| 9vHPV, 4vHPV or 2vHPV can be used for routine 	| yes                                          	|
| v1.9  	| raw paper text 	| FALSE 	| 64          	| DPR       	| 5             	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| yes                                           	| yes                                          	|
| v1.10 	| raw paper text 	| FALSE 	| 64          	| DPR       	| 10            	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| yes                                           	| yes                                          	|
| v1.11 	| raw paper text 	| FALSE 	| 64          	| DPR       	| 20            	| Flan-T5 Large 	| December 10, 2014                               	| 9vHPV, 4vHPV or 2vHPV can be used for routine               	| yes                                           	| yes                                          	|



### Summarizing the contexts 



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

5. Finally, run the RAG pipeline by specifying the parameters in a config file as follows: 
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