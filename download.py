# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import AutoTokenizer, pipeline
import torch
def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    pipeline = pipeline(
        "text-generation",
        model="tiiuae/falcon-7b-instruct",
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        # we pass model parameters here too
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=0.75,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=256,  # max number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating

    )

if __name__ == "__main__":
    download_model()