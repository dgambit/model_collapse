

from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from huggingface_hub import login


def gen_docs(gen, run, modelname, datasetname, ndoc, real, synt):

    print(f"Generation of epoch {gen} run {run} for {ndoc} documents")
    
    curr_model = f"dgambettavuw/M_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}_vuw"

    print(curr_model)

    prompt_url = f"dgambettavuw/P_{datasetname}_doc{ndoc}_real{real}"



    #HF_TOKEN AUTHENTICATION


    login(token = hf_token) #writeToken

    prompts = load_dataset(prompt_url, split="train")
    


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = curr_model,
        max_seq_length = 512,
        dtype = None,
        load_in_4bit = True,
        device_map = {"": 0})
    

    print(model)
    
    FastLanguageModel.for_inference(model)


    dict_docs = {"id": [],
                "doc": []}




    device = model.device  # Ottieni il dispositivo del modello

    
    for i in range(ndoc):
        
        dict_docs["id"].append(i)
        
        prompt = prompts["prompt"][i]
        
        
        input = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(input, max_new_tokens= synt, min_new_tokens=synt-8)
        doc = tokenizer.decode(output[0])
        
        doc = doc.replace("<s>", "")
        
        doc = doc.replace("<|begin_of_text|>","")

        doc = doc.replace("<|end_of_text|>", "")
        
        
        
        if i%100 == 0:
            print(i)
            print(doc)

        dict_docs["doc"].append(doc)


    new_dataset = Dataset.from_dict(dict_docs)

    new_dataset.push_to_hub(f"D_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}_vuw")



