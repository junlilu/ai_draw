import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

txt_file = []
for i in os.listdir("./data/sample/sample"):
    if i.split(".")[1] == "txt":
        txt_file.append("./data/sample/sample/{}".format(i))

tokenizer.train(txt_file, trainer)
tokenizer.save("./sample.json")
