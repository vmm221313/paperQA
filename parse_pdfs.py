import numpy as np 
import pandas as pd
from PyPDF2 import PdfReader

def main():
    title = []
    data = []
    for paper in glob.glob("data/papers_pdf/*.pdf"):
        title.append(paper.split("/")[-1][:-4])
        reader = PdfReader(paper)

        text = []
        for page_idx in range(len(reader.pages)):
            text.append(reader.pages[page_idx].extract_text())
        text = " ".join(text)
        text = re.sub("\n", "", text)
        
        data.append(text)

    for idx in range(len(data)):
        with open(f"data/papers_txt/{title[idx]}.txt", "w") as f:
            f.write(data[idx])


if __name__ == "__main__":
    main()