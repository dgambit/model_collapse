
from unsloth import FastLanguageModel

import os
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from unsloth import unsloth_train


from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

from huggingface_hub import login, HfApi, create_repo, Repository


def fine_tuning(gen, run, modelname, datasetname, ndoc, real, synt):
    
    print(f"Fine tuning of gen {gen} run {run}")
 
    curr_model = f"dgambettavuw/M_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}_vuw"

    
    curr_doc = f"dgambettavuw/D_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}_vuw"

    #HF_TOKEN AUTHENTICATION

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = curr_model, 
        max_seq_length = 2048,
        dtype = None,
       load_in_4bit = True,
        device_map = {"": 0}
    )


    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        return { "text" : [example + EOS_TOKEN for example in examples["doc"]] }
    
    dataset = load_dataset(curr_doc, split = "train")

    dataset = dataset.map(formatting_prompts_func, batched = True,)


    trainer = UnslothTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 8,

        args = UnslothTrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,

            warmup_ratio = 0.1,
            num_train_epochs = 5,

            learning_rate = 5e-5,
            embedding_learning_rate = 5e-6,

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.00,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
        ),
    )



    
    
    trainer_stats = trainer.train()

    directory = "./log_finetuning"
    os.makedirs(directory, exist_ok=True)

    
    file_path = f"{directory}/log_gen{gen}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}_vuw.txt"


    with open(file_path, "w") as file:
        file.write(str(trainer_stats))




    ############################################################################


    #FastLanguageModel.for_inference(model)

    next_model = f"M_gen{gen+1}_run{run}_{modelname}_{datasetname}_doc{ndoc}_real{real}_synt{synt}_vuw"


    repo_name = next_model
    api = HfApi()
    api.create_repo(repo_name, private=False)

    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)    


