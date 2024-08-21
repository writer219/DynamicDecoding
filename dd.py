import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random
import numpy as np

model_path = "./Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False,add_bos_token=True, add_eos_token = True,model_max_length=4096,padding_side="right",trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path,  torch_dtype=torch.bfloat16, device_map="auto",trust_remote_code=True,attn_implementation="flash_attention_2").eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_jsonl(file_path, instruction='question', output='answer'):
    data_list = []
    with open(file_path, 'r') as file:
        for line in file:
            json_obj = json.loads(line.strip())
            data_dict = {
                'instruction': json_obj[instruction],
                'output': json_obj[output]
            }
            data_list.append(data_dict)
    return data_list

fp = './test.jsonl'
list_data_dict = load_jsonl(fp, instruction='question', output='answer')
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer

def extract_answer(string):
    pattern = r'\[(.*?)\]'  
    matches = re.findall(pattern, string)
    if matches:
        last_match = matches[-1].replace(',', '')  
        numbers = re.findall(r'[\d.]+', last_match)
        if numbers:
            answer = numbers[-1]
            if answer.endswith('.'):
                answer = answer[:-1]  
            return answer
    return None


def find_indices_by_topp_and_topk(prob_list, topp, topk, n):
    prob_list_exp = [[np.exp(np.array(array)) for array in sublist] for sublist in prob_list]
    exp_sum = 0

    flat_list = []
    original_indices = []
    for i, sublist in enumerate(prob_list_exp):
        for j, array in enumerate(sublist):
            for k, value in enumerate(array):
                exp_sum += value
                flat_list.append(value)
                original_indices.append((i, j, k))

    probabilities = np.array(flat_list)

    sorted_indices = np.argsort(probabilities)[::-1]

    cumulative_sum = 0
    selected_indices = []
    selected_probabilities = []
    for idx in sorted_indices:
        cumulative_sum += probabilities[idx]
        selected_indices.append(original_indices[idx])
        selected_probabilities.append(probabilities[idx])
        if cumulative_sum > topp*exp_sum:
            break
    selected_length =  len(selected_probabilities)


    if len(selected_indices) > topk:
        selected_indices = selected_indices[:topk]
        selected_probabilities = selected_probabilities[:topk]

    grouped_indices = {}
    grouped_probabilities = {}
    
    for idx, prob in zip(selected_indices, selected_probabilities):
        first_two = (idx[0], idx[1])
        
        if first_two not in grouped_indices:
            grouped_indices[first_two] = []
            grouped_probabilities[first_two] = []
        
        grouped_indices[first_two].append(idx)
        grouped_probabilities[first_two].append(prob)
    

    sorted_grouped_indices = [grouped_indices[key] for key in sorted(grouped_indices.keys())]
    sorted_grouped_probabilities = [grouped_probabilities[key] for key in sorted(grouped_probabilities.keys())]
    
    for i in range(len(sorted_grouped_indices)):
        combined = list(zip(sorted_grouped_indices[i], sorted_grouped_probabilities[i]))
        combined.sort(key=lambda x: x[0][2])  
        sorted_grouped_indices[i], sorted_grouped_probabilities[i] = zip(*combined)
    
    original_selected_probabilities = []
    for group in sorted_grouped_indices:
        group_probs = []
        for idx in group:
            i, j, k = idx
            group_probs.append(prob_list[i][j][k])
        original_selected_probabilities.append(group_probs)

    return sorted_grouped_indices, original_selected_probabilities,selected_length

def group_indices_by_first_two(indices):
    grouped_indices = {}
    
    for sublist in indices:
        for idx in sublist:
            first_two = (idx[0], idx[1])
            third = idx[2]
            
            if first_two not in grouped_indices:
                grouped_indices[first_two] = []
            
            grouped_indices[first_two].append(torch.tensor([[third]]))
    
    for key in grouped_indices:
        grouped_indices[key].sort(key=lambda x: x.item())
    
    sorted_grouped_indices = [grouped_indices[key] for key in sorted(grouped_indices.keys())]
    
    return sorted_grouped_indices


def select_kv_by_indices(indices, original_list):
    flattened_indices = [idx for sublist in indices for idx in sublist]
    sorted_indices = sorted(flattened_indices, key=lambda x: (x[0], x[1]))
    
    selected_elements = []
    seen_indices = set()
    
    for idx in sorted_indices:
        first_two = (idx[0], idx[1])
        
        if first_two not in seen_indices:
            seen_indices.add(first_two)
            selected_elements.append(original_list[first_two[0]][first_two[1]])
    
    return selected_elements
def construct_new_branches(indices, branches,prompt_length):
    flattened_indices = [idx for sublist in indices for idx in sublist]
    sorted_indices = sorted(flattened_indices, key=lambda x: (x[0], x[1]))
    
    new_branches = {}
    
    for idx in sorted_indices:
        first_two = (idx[0], idx[1])
        third = idx[2]
        
        if first_two not in new_branches:
            new_branches[first_two] = []
        
        test = branches[first_two[0]][first_two[1]][0]
        new_branches[first_two].append(torch.cat([branches[first_two[0]][first_two[1]][0], torch.tensor([third]).cuda()],dim=0).unsqueeze(0))

    sorted_new_branches = [new_branches[key] for key in sorted(new_branches.keys())]
    
    return sorted_new_branches


def update_score_list(score_list, indice, prob_list):
    from collections import defaultdict

    updated_score_dict = defaultdict(list)

    for idx_group in indice:
        for idx in idx_group:
            i, j, k = idx  
            if i < len(score_list) and j < len(score_list[i]):
                prob_value = prob_list[i][j][k]
                new_score = score_list[i][j] + [prob_value]
                updated_score_dict[(i, j)].append(new_score)

  
    updated_score_list = []
    for (i, j), scores in updated_score_dict.items():
        updated_score_list.append(scores)

    return updated_score_list

def final_score(nested_list):

    result = []

    for sublist in nested_list:
        sublist_averages = []
        for array in sublist:
            if array:
                average = sum(array) / len(array)
            else:
                average = 0 
            sublist_averages.append(average)
        result.append(sublist_averages)

    return result

max_generation_steps = max_gene_tokens
threshold = topp
topk = topk

n_gram = 1
n_gram_step =0
final_answer_list = []
all_probs = []
selected_probs_all = []
index_list =[0]
max_prob_list = []
with torch.no_grad():
    for sample in list_data_dict:
        input_text = [
            {
                "role": "user", 
                "content": f"The following is a math problem from elementary school. Please provide the solution process and the final result. Write the final result within square brackets. Only one pair of square brackets should be used, and it should contain only a number. \n The question is: {sample['instruction']}"
            }
            ]
        tokenized_chat = tokenizer.apply_chat_template(input_text, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
        temp_input_ids = [[tokenized_chat]]
        temp_branches = [[tokenized_chat]]
        branches = [[tokenized_chat]]
        prompt_length = len(branches[0][0][0])
        temp_kv = [None]
        selected_probabilities = [[0]]
        test_score = [[[]]]
        answers = []
        branch_list = []
        input_ids_list=temp_input_ids
        kvcache = temp_kv
        previous_score=[[[]]]
        
        select_length_list = []
        for step in range(max_generation_steps):

            score = []
            temp_selectpath_probs = []
            probs_list = []
            all_kv = []
            original_probs_list = []
            original_probs_list_nolog = []
            count = 0
            all_temp_score = []
            temp_branches_list = []
            should_break = False
            all_path = 0
            else_path = 0
            if_path = 0
            temp_max_prob = []
            for k in range(len(input_ids_list)):
                temp_probs_list = []
                temp_kv = []
                temp_original_probs = []
                temp_original_probs_nolog = []
                temp_score = []
                temp_branches = []
            
                for j in range(len(input_ids_list[k])):
      
                    if input_ids_list[k][j][-1][-1] == tokenizer.eos_token_id:
                        if_path += 1
                        all_path += 1
                        answer_sentence = tokenizer.decode(branches[k][j][0][len(tokenized_chat[0]):],skip_special_tokens=True)
                        model_answer = extract_answer(answer_sentence)

                        is_cor = is_correct(model_answer, sample['output'])
                        answer = {
                            "model_answer": f"{answer_sentence}",
                            "is_cor": f"{is_cor}",
                            "score": f"{np.mean(test_score[k][j])}"
                        }
                        answers.append(answer)
                    else:
                        else_path += 1
                        all_path += 1
                        outputs = model(input_ids = input_ids_list[k][j].cuda(), past_key_values = kvcache[k])
                        count += 1
                        logits = outputs.logits[:,-1]
                        temp_kv.append(outputs.past_key_values)

                        logprobs_origin = torch.nn.functional.log_softmax(logits, dim=-1)[0].cpu().numpy()
                        probs_origin = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()

                        logprobs = logprobs_origin+selected_probabilities[k][j]
                        temp_max_prob.append(np.exp(max(logprobs)))
        
                        temp_probs_list.append(logprobs)
                        temp_original_probs.append(logprobs_origin)
                        temp_original_probs_nolog.append(probs_origin)
                        temp_score.append(test_score[k][j])
                        temp_branches.append(branches[k][j])
                all_temp_score.append(temp_score)
                probs_list.append(temp_probs_list)
                original_probs_list.append(temp_original_probs)
                original_probs_list_nolog.append(temp_original_probs_nolog)
                all_kv.append(temp_kv)
                temp_branches_list.append(temp_branches)
            if all_path == if_path:
                break

            indice, selected_probabilities,selected_length = find_indices_by_topp_and_topk(probs_list, topp = threshold, topk=topk, n=step+1)
            select_length_list.append(selected_length)
            test_score = update_score_list(all_temp_score,indice,original_probs_list_nolog)
            input_ids_list = group_indices_by_first_two(indice)
            kvcache = select_kv_by_indices(indice, all_kv)
            branches = construct_new_branches(indice, temp_branches_list,prompt_length)
            avg_scores = final_score(test_score)


            branch_list.append(count)

            max_prob_list.append(max(temp_max_prob))
        total_count = len(answers)
        if total_count == 0:
            accuracy = 0
            max_is_cor = False
        else:
            max_score_answer = max(answers, key=lambda x: float(x['score']))
            max_is_cor = max_score_answer['is_cor']
            correct_count = sum(1 for item in answers if item['is_cor'] == 'True')
            accuracy = correct_count / total_count
        sample_answer = {
            'question': sample['instruction'],
            'average_beam': f"{np.mean(branch_list)}",
            'num_output':f'{len(answers)}',
            'acc rate':f'{accuracy}',
            'max is cor':f'{max_is_cor}',
            'total token':f'{np.sum(branch_list)}'
            }   
