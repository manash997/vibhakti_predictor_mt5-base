from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from peft import PeftModel, PeftConfig
peft_model_id = "manashxml/my_awsm_mlm_mt5-base-50"
model = AutoModelForSeq2SeqLM.from_pretrained("my-base-model")
model=PeftModel.from_pretrained(model,"my-awesome-model")
tokenizer = AutoTokenizer.from_pretrained("my-awesome-tokenizer")

import torch
#model = model.to(device)
def gen_op(sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=128)
        return (tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])
def gen_sen(sentence,op_tokens):
    sent_list=sentence.strip().split()
    count=0
    op_tok_list=op_tokens.split()
    for word_inx in range(len(sent_list)):
        if sent_list[word_inx]=="[MASK]":
            sent_list[word_inx]=op_tok_list[count]
            count+=1
    return " ".join(sent_list)
            
def gen_vibhakti_prediction(sentence):
    sent_list=sentence.strip().split()
    count=0
    for word in sent_list:
        if word=="[MASK]":
            count+=1
    if count<3:
        #print(line)
        op=gen_op(sentence.strip())
        op=gen_sen(sentence,op)
        return op