# from sacrebleu import sentence_bleu

# refs = []
# hyps = []

# with open("ab.txt", "rb") as file:
#     lines = file.read().decode('utf-8', 'ignore').splitlines()
# num_sentences = len(lines) // 4

# for i in range(num_sentences):
#     src_sent = lines[i * 4].strip().split(":")[1].strip()
#     tgt_sent = lines[i * 4 + 1].strip().split(":")[1].strip()
#     pred_sent = lines[i * 4 + 2].strip().split(":")[1].strip()
#     refs.append(tgt_sent)
#     bleu = sentence_bleu(pred_sent, refs)
#     bleu_score = bleu.score
#     hyps.append(bleu_score)

# average_bleu = sum(hyps) / num_sentences
# print(f"Average BLEU score: {average_bleu}")
from sacrebleu import corpus_bleu

refs = []
hyps = []

with open("ab.txt", "r") as file:
    lines = file.readlines()

for i in range(0, len(lines), 4):
    src_sent = lines[i].strip().split(":")[1].strip()
    tgt_sent = lines[i + 1].strip().split(":")[1].strip()
    pred_sent = lines[i + 2].strip().split(":")[1].strip()
    refs.append([tgt_sent])
    hyps.append(pred_sent)

bleu = corpus_bleu(hyps, [refs])
print(f"BLEU score: {bleu.score}")
