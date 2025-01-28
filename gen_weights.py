from datasets import Dataset, load_dataset

#dataset = load_dataset("dgambettaphd/wikitext103", split = "train")


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import json
import os

#colors 
# = plt.cm.jet(np.linspace(0,1,11))

nsent = 1000



for dataset in ["xlsum", "sciabs"]:
    for task in ["wiki", "xlsum", "sciabs"]:

        N=11

        ds = load_dataset(f"dgambettaphd/{task}", split = "train")
        sents = ds["text"][1000:1000+nsent]
        sents = [" ".join(s.split(" ")[:25]) for s in sents]
        

        for run in [0]:

            file_path = f"stats_vuw/next_tok_probs/pipeline_{dataset}/task_{task}_{nsent}/"
            
            print("Generating..."+ file_path + f'run_{run}.json')

            top100_dict = {32: {},
                        64: {},
                        96: {}}

            for synt in [32, 64, 96]:

                for gen in range(N):

                    top100_dict[synt][gen] = []

                    model_name = f"dgambettavuw/M_gen{gen}_run{run}_llama2-7b_{dataset}_doc1000_real{128-synt}_synt{synt}_vuw"  # Replace with your model

                    #model_name = f"danigambit/M_gen{gen}_bench"
                    
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    device = model.device  # Ottieni il dispositivo del modello


                    for input_text in sents:

                        with torch.no_grad():
                            inputs = tokenizer(input_text, return_tensors="pt").to(device)

                            outputs = model(**inputs)
                            logits = outputs.logits

                            last_token_logits = logits[0, -1, :]
                            probs = F.softmax(last_token_logits, dim=-1)

                            top100 = ((torch.topk(probs, 100).values).tolist(), [tokenizer.decode(t) for t in torch.topk(probs, 100).indices])
                            
                            top100_dict[synt][gen].append(top100)

                    #[tokenizer.decode(t) for t in torch.topk(probs, 10).indices]
                    

            '''
            os.makedirs(f"stats_vuw/next_tok_probs/pipeline_{dataset}/task_{task}", exist_ok=True)

            #with open(f'weights/llama2-7b_wiki_doc1000/1000sent/{run}.json', 'w') as fp:
            with open(f'stats_vuw/next_tok_probs/pipeline_{dataset}/task_{task}/{run}.json', 'w') as fp:
                json.dump(top100_dict, fp)
            '''


            os.makedirs(file_path , exist_ok=True)

            print("Saving..."+ file_path + f'run_{run}.json')
            with open(file_path + f'run_{run}.json', 'w') as fp:
                json.dump(top100_dict, fp)
