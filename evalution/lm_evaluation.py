import os

import sampler.lm_tokenizer as lm_tokenizer
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import torch
import pandas as pd
import numpy as np
import micro_lm_model  
from transformers import PreTrainedTokenizerFast
from torch.amp import autocast

subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}

PROMPT_DICT = {
    "choice": (
        "The following are multiple-choice questions, please choose the correct answer.\n"
        "{passage}\n"
        "{question}\n"
        "A: {A}\n"
        "B: {B}\n"
        "C: {C}\n"
        "D: {D}\n"
        "Response:"
    ),
    "choice2": (
        "{question}\n"
        "A: {A}\n"
        "B: {B}\n"
        "C: {C}\n"
        "D: {D}\n"
        "Response:"
    )
}

def convert_and_infer(model, prompt, tokenizer: PreTrainedTokenizerFast, config, device, valid_outputs):
    inputs = tokenizer.encode(prompt)
    inputs = torch.from_numpy(np.array([inputs])).to(torch.long)
    is_trunc = False
    if inputs.shape[1] > config.sequence_length:
        inputs = inputs[:, -config.sequence_length:]
        is_trunc = True
    inputs = inputs.to(device)

    with autocast(dtype=torch.bfloat16):
        logits = model(inputs)
        logits = logits[:, -1].cpu()
        valid_probs = logits[0][valid_outputs]
        next_token = torch.argmax(valid_probs).item()
        result = tokenizer.decode(valid_outputs[next_token])
    return result, is_trunc

def multi_choice_eval(name, model, tokenizer, data, config, device, valid_response, response_dict, prompt_keys):
    st = time.time()
    valid_outputs = [tokenizer.encode(res)[-1] for res in valid_response]

    knt_trunc = 0
    knt_correct = 0
    knt_total = 0
    for i, sample in data.iterrows():
        dmap = {
            'passage': sample[prompt_keys[0]] if prompt_keys[0] in sample else prompt_keys[0],
            'question': sample[prompt_keys[1]],
            'A': sample[prompt_keys[2]] if prompt_keys[2] in sample else prompt_keys[2],
            'B': sample[prompt_keys[3]] if prompt_keys[3] in sample else prompt_keys[3],
            'C': sample[prompt_keys[4]] if prompt_keys[4] in sample else prompt_keys[4],
            'D': sample[prompt_keys[5]] if prompt_keys[5] in sample else prompt_keys[5],
        }
        prompt = PROMPT_DICT['choice'].format_map(dmap)
        label = sample[prompt_keys[6]]
        label = response_dict[label]

        result, is_trunc = convert_and_infer(model, prompt, tokenizer, config, device, valid_outputs)
        if is_trunc: knt_trunc += 1
        if label == result: knt_correct += 1
        knt_total += 1

    print(f"{name} eval trucs: {knt_trunc}, correct: {knt_correct}, total: {knt_total}, ratio: {knt_correct/knt_total:.2f}, consume: {time.time() - st:.1f}s")

def common_sense_reasoning(model, tokenizer, data_root, config, device):
    # boolq eval trucs: 1, correct: 2033, total: 3270, ratio: 0.62, consume: 26.4s
    data_path = "boolq/validation/0000.parquet"
    data = pd.read_parquet(os.path.join(data_root, data_path))
    valid_response = ['A', 'B']
    response_dict = {True: 'A', False: 'B'}
    prompt_keys = ['passage', 'question', 'True', 'False', 'None', 'None', 'answer']
    multi_choice_eval("boolq", model, tokenizer, data, config, device, valid_response, response_dict, prompt_keys)

    # piqa eval trucs: 0, correct: 912, total: 1838, ratio: 0.50, consume: 5.2s
    data_path = "piqa/validation/0000.parquet"
    data = pd.read_parquet(os.path.join(data_root, data_path))
    valid_response = ['A', 'B']
    response_dict = {0: 'A', 1: 'B'}
    prompt_keys = ['', 'goal', 'sol1', 'sol2', 'None', 'None', 'label']
    multi_choice_eval("piqa", model, tokenizer, data, config, device, valid_response, response_dict, prompt_keys)

    # hellaswag eval trucs: 0, correct: 2513, total: 10042, ratio: 0.25, consume: 45.4s
    data_path = "hellaswag/validation/0000.parquet"
    data = pd.read_parquet(os.path.join(data_root, data_path))
    for i in range(4):
        data[f'sol{i+1}'] = data['endings'].apply(lambda x: x[i])
    valid_response = ['A', 'B', 'C', 'D']
    response_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D'}
    prompt_keys = ['', 'ctx', 'sol1', 'sol2', 'sol3', 'sol4', 'label']
    multi_choice_eval("hellaswag", model, tokenizer, data, config, device, valid_response, response_dict, prompt_keys)

def gen_prompt(data, subject, k):
    subject = ' '.join(subject.split('_'))
    # prompt = "The following are multiple-choice questions, please choose the correct answer.\n"
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(subject)
    for i, sample in data.iterrows():
        if i >= k: break
        prompt += PROMPT_DICT['choice2'].format_map(sample) + ' ' + sample['Answer'] + '\n'
    return prompt

def mmlu(model, tokenizer, data_root, config, device, k_shot=0):
    dev_dir = "mmlu/data/dev"
    test_dir = "mmlu/data/test"
    postfix = ".csv"
    columns = ["question", "A", "B", "C", "D", "Answer"]
    valid_response = ['A', 'B', 'C', 'D']
    valid_outputs = [tokenizer.encode(res)[-1] for res in valid_response]
    print(valid_outputs)

    cate2cls = {}
    for cls, cates in categories.items():
        for cate in cates:
            cate2cls[cate] = cls

    subcate_scores = {}
    cls_scores = {}
    for sub_cate, cate in subcategories.items():
        cate = cate[0]
        cls = cate2cls[cate]
        if cls not in cls_scores:
            cls_scores[cls] = []

        dev_file = os.path.join(data_root, dev_dir, sub_cate + '_dev' + postfix)
        dev_data = pd.read_csv(dev_file, header=None)
        dev_data.columns = columns

        test_file = os.path.join(data_root, test_dir, sub_cate + '_test' + postfix)
        test_data = pd.read_csv(test_file, header=None)
        test_data.columns = columns

        knt_correct = 0
        knt_total = 0
        knt_trunc = 0
        dev_prompt = gen_prompt(dev_data, sub_cate, k_shot)
        for i, sample in test_data.iterrows():
            example = PROMPT_DICT['choice2'].format_map(sample)
            label = sample['Answer']

            k = k_shot
            prompt = dev_prompt + example
            while len(tokenizer.encode(prompt)) > config.sequence_length and k > 0:
                k -= 1
                train_prompt = gen_prompt(dev_data, sub_cate, k)
                prompt = train_prompt + example

            result, is_trunc = convert_and_infer(model, prompt, tokenizer, config, device, valid_outputs)
            result = result.strip()
            if is_trunc: knt_trunc += 1
            if label == result: knt_correct += 1
            knt_total += 1

        score = knt_correct / knt_total
        subcate_scores[sub_cate] = score
        cls_scores[cls].append(score)
        print(f"{sub_cate} correct: {knt_correct}, total: {knt_total}, ratio: {score:.2f}, trunc: {knt_trunc}")

    for cls, scores in cls_scores.items():
        print(f"{cls}: {np.mean(scores):.2f}")

def triviaQA(model, tokenizer, data_root, config, device):
    data_path = 'triviaqa/data/validation-00000-of-00001-e0b96d7de24c1bf5.parquet'
    data = pd.read_parquet(os.path.join(data_root, data_path))
    pass

def closed_book_qa(model, tokenizer, data_root, config, device):
    # TriviaQA
    triviaQA(model, tokenizer, data_root, config, device)
    # NaturalQuestions
    pass

def reading_comprehension(model, tokenizer, data_root, config, device):
    valid_response = ['A', 'B', 'C', 'D']
    response_dict = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D'}
    prompt_keys = ['article', 'question', 'sol1', 'sol2', 'sol3', 'sol4', 'answer']

    # race-middle eval trucs: 0, correct: 306, total: 1436, ratio: 0.21, consume: 25.6s
    data_path = 'race/middle_test.parquet'
    data = pd.read_parquet(os.path.join(data_root, data_path))
    for i in range(4):
        data[f'sol{i+1}'] = data['options'].apply(lambda x: x[i])
    multi_choice_eval("race-middle", model, tokenizer, data, config, device, valid_response, response_dict, prompt_keys)

    # race-high eval trucs: 27, correct: 757, total: 3498, ratio: 0.22, consume: 64.5s
    data_path = 'race/high_test.parquet'
    data = pd.read_parquet(os.path.join(data_root, data_path))
    for i in range(4):
        data[f'sol{i+1}'] = data['options'].apply(lambda x: x[i])
    multi_choice_eval("race-high", model, tokenizer, data, config, device, valid_response, response_dict, prompt_keys)

def main(model_path: str, tokenizer_path: str = None, data_root: str=None):
    model = micro_lm_model.load_model(model_path)
    model = model.eval()
    tokenizer = lm_tokenizer.load_tokenizer()

    with torch.no_grad():
        common_sense_reasoning(model, tokenizer, data_root, )
        mmlu(model, tokenizer, data_root, config, device, k_shot=0)
        # closed_book_qa(model, tokenizer, data_root, config, device)
        reading_comprehension(model, tokenizer, data_root, config, device)

if __name__ == "__main__":
    main()