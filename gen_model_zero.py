
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from huggingface_hub import login
from huggingface_hub import login, HfApi


def gen_model_zero(run, modelname, datasetname, ndoc, real, synt):

    base_model = f"dgambettavuw/M_{modelname}"

    hf_token = "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"
    login(token = hf_token) 

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = base_model,
            max_seq_length = 512,
            dtype = None,
            load_in_4bit = True,
            device_map = {"": 0})

    #FastLanguageModel.for_inference(model)

    print(f"PIPELINE:{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}")

    zero_model = f"dgambettavuw/M_gen0_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}_vuw"

    repo_name = zero_model
    api = HfApi()
    api.create_repo(repo_name, private=False)

    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)    
