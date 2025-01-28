from huggingface_hub import login
from unsloth import FastLanguageModel
from huggingface_hub import login, HfApi


import fine_tuning
import gen_docs
import gen_model_zero

modelname="llama2-7b"
datasetname="wiki"
ndoc = 1000
run=0

for real in [32, 64, 96]:

        #ntokI = 32, 64, 96
        synt = 128 - real

        #fun(ep, run, modelname, datasetname, ndoc, ntok)

        gen_model_zero.gen_model_zero(run, modelname, datasetname, ndoc, real, synt)
        gen_docs.gen_docs(0, run, modelname, datasetname, ndoc, real, synt) 

        for gen in range(10):

                print(gen, run, modelname, datasetname, ndoc,  real, synt)

                fine_tuning.fine_tuning(gen, run, modelname, datasetname, ndoc, real, synt) 
                gen_docs.gen_docs(gen+1, run, modelname, datasetname, ndoc, real, synt) 


