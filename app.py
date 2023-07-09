from potassium import Potassium, Request, Response

import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import sys


app = Potassium("my_app")
sys.path.append("/")
def summarize_thought(x, llm):

  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

  docs = text_splitter.create_documents([x])

  chain = load_summarize_chain(llm, chain_type="refine")

  try:
    resp = chain.run(docs)

  except Exception as e:

    print(e)

    try:

      resp = chain.run(docs)

    except Exception as e:

      print(e)
      resp = "-1"

    return resp

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = "tiiuae/falcon-7b-instruct"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
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
    
    llm = HuggingFacePipeline(pipeline=pipeline)

    # model = pipeline('fill-mask', model='bert-base-uncased', device=device)
   
    context = {
        "model": llm
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")

    model = context.get("model")

    outputs = summarize_thought(prompt, model)
    return Response(
        json = {"outputs": outputs[0]}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
