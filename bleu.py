import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from tqdm import tqdm

try:
    references = []
    hypotheses = []
    with open("lay4.txt", "r") as file:
        lines = file.readlines()
        for i in tqdm(range(0, len(lines), 4), desc="Computing BLEU scores"): 
            if i+2 >= len(lines):
                continue
            target_line = lines[i+1].strip()
            predict_line = lines[i+2].strip()

            target_tokens = word_tokenize(target_line[len("Target: "):])
            predict_tokens = word_tokenize(predict_line[len("Predict: "):])
            references.append([target_tokens])
            hypotheses.append(predict_tokens)

    bleu_score = corpus_bleu(references, hypotheses)
    print(f"BLEU score: {bleu_score}")
except Exception as e:
    print(f"An error occurred: {e}")
