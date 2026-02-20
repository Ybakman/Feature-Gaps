import os
#set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
import TruthTorchLM as ttlm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import pickle
import tqdm
import re
import argparse
import numpy as np
from sklearn.decomposition import PCA
import copy
import argparse
from perturb_utils import *


args = argparse.ArgumentParser()
args.add_argument("--model_name", type=str, default= 'meta-llama_Llama-3.1-8B-Instruct')
args.add_argument("--model_path", type=str, default= '../../model-registry')


args.add_argument("--data_path", type=str, default= '../results')
args.add_argument("--data_name", type=str, default= 'qasper_benchmark8b_full.pkl')


args.add_argument("--easy_context_path", type=str, default= '../easy_contexts')
args.add_argument("--easy_context_name", type=str, default= 'llama8b_qasper_benchmark8b_full.pkl')


args.add_argument("--save_path", type=str, default= '../hidden_states')
args.add_argument("--save_name", type=str, default= 'llama8b')

args.add_argument("--prompt", type=str, default= 'regular')
args.add_argument('--perturb_modes', type=str, default=['regular'], nargs='*', help='list of perturbations',)

args.add_argument("--sample_num", type=int, default= -1)#-1 means use all samples

args.add_argument('--store_probs',  action='store_true')
args.add_argument('--all_states',  action='store_true')

args = args.parse_args()

all_states = args.all_states

data_path = args.data_path
data_name = args.data_name
data_dir = os.path.join(data_path, data_name)
with open(data_dir, 'rb') as file:
    dataset = pickle.load(file)
    print(f"Data successfully loaded from {data_dir}:")

perturb_modes = args.perturb_modes
print(perturb_modes)
easy_context_found = False
for perturb_mode in perturb_modes:
    if 'easy_context' in perturb_mode:
        easy_context_found = True

if easy_context_found == True:
    easy_context_path = args.easy_context_path
    easy_context_name = args.easy_context_name
    easy_context_dir = os.path.join(easy_context_path, easy_context_name )
    with open(easy_context_dir, 'rb') as file:
        easy_contexts = pickle.load(file)

prompt = args.prompt

#load the model
model_path = args.model_path
model_name = args.model_name
#model_name = "meta-llama_llama-3.2-1b-instruct"
#model_dir = os.path.join(model_path, model_name)
model_dir = model_name
# max_memory_mapping = {
#                 0: "40GiB",     
#                 1: "40GiB",
#                 2: "40GiB",
#                 3: "40GiB",
#                 4: "40GiB",
#                 5: "40GiB",
#                 6: "40GiB",
#                 7: "40GiB",
              
#             }
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto")#, max_memory=max_memory_mapping
tokenizer = AutoTokenizer.from_pretrained(model_dir)

if prompt == 'regular':
    prompt_positive_template = "Context: {context}. Here is a given context. You are a helpful assistant. Answer the following question with a brief single but complete answer. Use the context information to answer this question. Question: {question}. Answer:"
    prompt_negative_template  = "Context: {context}. Here is a given context. You are a helpful assistant. Answer the following question with a brief single but complete answer. Use the context information to answer this question. Question: {question}. Answer:"
elif prompt == 'context':
    prompt_positive_template = "Context: {context}. \n Here is a given context. You are a helpful assistant. Answer the following question with a brief single but complete answer. Use the context information to answer this question. Do not use own knowledge. Just look at the context. Question: {question} Answer:"
    prompt_negative_template  = "Context: {context}. \n Here is a given context. You are a helpful assistant. Answer the following question with a brief single but complete answer. DO NOT use the context information to answer this question. Use your own knowledge. Ignore the context. Question: {question} Answer:"
elif prompt == 'honesty':
    prompt_positive_template  = "Context: {context}. \n Here is a given context. You are a helpful assistant. Answer the following question with a brief single but complete answer. Be honest with your answer. Do not give unfactual information. Do not lie. Question: {question} Answer:"
    prompt_negative_template  = "Context: {context}. \n Here is a given context. You are a helpful assistant. Answer the following question with a brief single but complete answer. Be liar. Give an unfactual information. Tell a lie. Question: {question} Answer:"



sample_num = args.sample_num

questions = dataset['question_text']
contexts = dataset['contexts']
generation_dicts = dataset['generation_dicts']
correctness = dataset['generations_correctness']



if sample_num != -1:
    #select by looking at the context length
    count = 0
    questions = []
    contexts = []
    generation_dicts = []
    correctness = []
    index = 0
    if easy_context_found: 
        easy_contexts_old = easy_contexts
        easy_contexts = []
    while count < sample_num and index < len(dataset['contexts']):
        if len(dataset['contexts'][index]) > 85000:
            index += 1
            continue
        questions.append(dataset['question_text'][index])
        contexts.append(dataset['contexts'][index])
        generation_dicts.append(dataset['generation_dicts'][index])
        correctness.append(dataset['generations_correctness'][index])
        if easy_context_found: 
            easy_contexts.append(easy_contexts_old[index])
        index += 1
        count += 1
    # questions = questions[:sample_num]
    # contexts = contexts[:sample_num]
    # generation_dicts = generation_dicts[:sample_num]
    # correctness = correctness[:sample_num]
    print(f"number of samples: {len(questions)}")
    sample_num = len(questions)
else:
    sample_num = len(questions)


regular_hidden_states_avg = {}
perturbed_hidden_states_avg = {}
regular_hidden_states_first = {}
perturbed_hidden_states_first = {}
regular_hidden_states_all = {}
perturbed_hidden_states_all = {}
for perturb_mode in perturb_modes:
    perturbed_hidden_states_avg[perturb_mode] = {}
    perturbed_hidden_states_first[perturb_mode] = {}
    perturbed_hidden_states_all[perturb_mode] = {}

topk_indices = []
topk_values = []
perturbed_contexts = []
for i in tqdm.tqdm(range(sample_num)):
    with torch.no_grad():
        question = questions[i]
        context = contexts[i]
        all_ids = generation_dicts[i]['model_output'].to(model.device)
        text = generation_dicts[i]['text']
        input_ids = torch.tensor(tokenizer.encode(text))
        input_ids = input_ids.reshape(1,-1)
        prompt_positive = prompt_positive_template.format(context=context, question=question)
        messages_positive = [{"role": "user", "content": f"{prompt_positive}"}]
        input_text_positive = tokenizer.apply_chat_template(messages_positive, tokenize= False, add_generation_prompt = True, continue_final_message = False)
        input_ids_positive = tokenizer(input_text_positive, return_tensors="pt")['input_ids'].to(model.device)
        #append the generation
        generation_ids =  all_ids[:,len(input_ids[0]):]
        new_input_ids = torch.cat((input_ids_positive, generation_ids), dim=1).to(model.device)
        
        output_positive = model(new_input_ids, output_hidden_states = True, return_dict = True)

        if args.store_probs == True:
            logits = output_positive['logits'][0, len(input_ids_positive[0])-1:-1]
            topk = torch.topk(logits,100)
            topk_values.append(topk.values.cpu())
            topk_indices.append(topk.indices.cpu())
        hidden_states_positive = output_positive['hidden_states']
        for l in range(len(hidden_states_positive)):
            if l not in regular_hidden_states_avg:
                regular_hidden_states_avg[l] =  [hidden_states_positive[l][0, len(input_ids_positive[0])-1:-1, :].mean(axis=0).cpu().detach().numpy()]
                regular_hidden_states_first[l] =  [hidden_states_positive[l][0, len(input_ids_positive[0])-1, :].cpu().detach().numpy()]
                if all_states == True:
                    regular_hidden_states_all[l] =  [hidden_states_positive[l][0, len(input_ids_positive[0])-1:-1, :].cpu().detach().numpy()]
            else:
                regular_hidden_states_avg[l].append(hidden_states_positive[l][0, len(input_ids_positive[0])-1:-1, :].mean(axis=0).cpu().detach().numpy())
                regular_hidden_states_first[l].append(hidden_states_positive[l][0, len(input_ids_positive[0])-1, :].cpu().detach().numpy())
                if all_states == True:
                    regular_hidden_states_all[l].append(hidden_states_positive[l][0, len(input_ids_positive[0])-1:-1, :].cpu().detach().numpy())


        output_positive = None
        hidden_states_positive = None

        for j, perturb_mode in enumerate(perturb_modes):
            if "regular" in perturb_mode:
                perturbed_context = context #No change
            if  "sentence_order" in perturb_mode:
                perturbed_context = attack_coherence_sentence_order(copy.copy(context))
            if "mask_word" in perturb_mode:
                perturbed_context = mask_text(copy.copy(context), 15)
            if 'typo' in perturb_mode:
                perturbed_context = attack_grammar_typos(copy.copy(context))
            if "word_order" in perturb_mode:
                perturbed_context = attack_grammar_word_order(copy.copy(context))
            if "easy_context" in perturb_mode:
                perturbed_context =  context + " " + easy_contexts[i]
            

            perturbed_contexts.append(perturbed_context)
            prompt_perturbed_context = prompt_negative_template.format(context=perturbed_context, question=question)

            messages_perturbed_context = [{"role": "user", "content": f"{prompt_perturbed_context}"}]
            input_text_perturbed_context = tokenizer.apply_chat_template(messages_perturbed_context, tokenize= False, add_generation_prompt = True, continue_final_message = False)
            input_ids_perturbed_context = tokenizer(input_text_perturbed_context, return_tensors="pt")['input_ids'].to(model.device)

            new_input_ids_perturbed = torch.cat((input_ids_perturbed_context, generation_ids), dim=1).to(model.device)#append the generation

            output_perturbed_context = model(new_input_ids_perturbed, output_hidden_states = True, return_dict = True)
            hidden_states_perturbed_context = output_perturbed_context['hidden_states']
            for l in range(len(hidden_states_perturbed_context)):
                if l not in perturbed_hidden_states_avg[perturb_mode]:
                    perturbed_hidden_states_avg[perturb_mode][l] =  [hidden_states_perturbed_context[l][0, len(input_ids_perturbed_context[0])-1:-1, :].mean(axis=0).cpu().detach().numpy()]
                    perturbed_hidden_states_first[perturb_mode][l] =  [hidden_states_perturbed_context[l][0, len(input_ids_perturbed_context[0])-1, :].cpu().detach().numpy()]
                    if all_states == True:
                        perturbed_hidden_states_all[perturb_mode][l] =  [hidden_states_perturbed_context[l][0, len(input_ids_perturbed_context[0])-1:-1, :].cpu().detach().numpy()]

                else:
                    perturbed_hidden_states_avg[perturb_mode][l].append(hidden_states_perturbed_context[l][0, len(input_ids_perturbed_context[0])-1:-1, :].mean(axis=0).cpu().detach().numpy())
                    perturbed_hidden_states_first[perturb_mode][l].append(hidden_states_perturbed_context[l][0, len(input_ids_perturbed_context[0])-1, :].cpu().detach().numpy())
                    if all_states == True:
                        perturbed_hidden_states_all[perturb_mode][l].append(hidden_states_perturbed_context[l][0, len(input_ids_perturbed_context[0])-1:-1, :].cpu().detach().numpy())
        
            hidden_states_perturbed_context = None
            output_perturbed_context = None


results = {}
results['prompt_positive'] = prompt_positive_template
results['prompt_negative'] = prompt_negative_template
results['questions'] = questions
results['correctness'] = correctness
results['perturb_modes'] = perturb_modes
results['contexts'] = contexts
results['perturbed_contexts'] = perturbed_contexts
results['hidden_states_positive_avg'] = regular_hidden_states_avg 
results['hidden_states_negative_avg'] = perturbed_hidden_states_avg
results['hidden_states_positive_first'] = regular_hidden_states_first 
results['hidden_states_negative_first'] = perturbed_hidden_states_first
results['hidden_states_positive_all'] = regular_hidden_states_all 
results['hidden_states_negative_all'] = perturbed_hidden_states_all
results['generation_dicts'] = generation_dicts 
results['topk_indices'] = topk_indices
results['topk_values']  = topk_values


save_path = args.save_path
save_name = f'{args.save_name}_{args.data_name}_{args.prompt}_{args.perturb_modes}'
save_dir = os.path.join(save_path, save_name)
with open(f'{save_dir}', 'wb') as f:
    pickle.dump(results , f)
    print(f'Results Saved to {save_dir}')




