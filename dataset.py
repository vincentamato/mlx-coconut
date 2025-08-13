import mlx.core as mx
import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

class DataLoader:
    def __init__(self, dataset, batch_size, collator, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collator = collator
        self.shuffle = shuffle
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collator(batch)
    
    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

def get_dataset(path, tokenizer, max_size=1000000000):
    def tokenize_sample(sample):
        question_text = sample["question"] + "\n"
        question_tokenized = [tokenizer.eot_token] + tokenizer.encode(question_text)
        steps_tokenized = [tokenizer.encode(s + "\n") for s in sample["steps"]]
        answer_tokenized = tokenizer.encode("### " + sample["answer"]) + [tokenizer.eot_token]

        return {
            "question_tokenized": question_tokenized,
            "steps_tokenized": steps_tokenized,
            "answer_tokenized": answer_tokenized,
            "idx": sample["idx"],
        }

    data = json.load(open(path))[:max_size]
    data = [{**d, "idx": idx} for idx, d in enumerate(data)]

    processed_data = []
    print(f"Processing {len(data)} samples...")
    
    for sample in data:
        processed_sample = tokenize_sample(sample)
        processed_data.append(processed_sample)

    if len(data) > 0:
        d = data[0]
        complete = d["question"] + "\n" + "\n".join(d["steps"]) + "\n### " + d["answer"]
        complete_tokenized = [tokenizer.eot_token] + tokenizer.encode(complete) + [tokenizer.eot_token]
        
        expected = (
            processed_data[0]["question_tokenized"]
            + list(itertools.chain.from_iterable(processed_data[0]["steps_tokenized"]))
            + processed_data[0]["answer_tokenized"]
        )
        
        assert complete_tokenized == expected, "Tokenization verification failed"
        print("Dataset tokenization verified successfully")

    return processed_data


@dataclass
class Collator:
    tokenizer: Any
    latent_id: Optional[int] = None
    label_pad_token_id: Optional[int] = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, mx.array]:
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', self.tokenizer.eot_token)

        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)
            
            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(self.latent_id)
                else:
                    n_tok_pad = 0
                
                feature["position_ids"] = [0] * n_tok_pad + list(range(len(feature["input_ids"])))
                feature["input_ids"] = [pad_token_id] * n_tok_pad + feature["input_ids"]
                
                if "labels" in feature:
                    feature["labels"] = [self.label_pad_token_id] * n_tok_pad + feature["labels"]
                
                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {}
        
        batch["input_ids"] = []
        for feature in features:
            padded = feature["input_ids"] + [pad_token_id] * (max_length - len(feature["input_ids"]))
            batch["input_ids"].append(padded)
        
        batch["attention_mask"] = []
        for feature in features:
            padded = feature["attention_mask"] + [0] * (max_length - len(feature["attention_mask"]))
            batch["attention_mask"].append(padded)
        
        if "labels" in features[0]:
            batch["labels"] = []
            for feature in features:
                padded = feature["labels"] + [self.label_pad_token_id] * (max_length - len(feature["labels"]))
                batch["labels"].append(padded)
        
        if "position_ids" in features[0]:
            batch["position_ids"] = []
            for feature in features:
                padded = feature["position_ids"] + [0] * (max_length - len(feature["position_ids"]))
                batch["position_ids"].append(padded)
        
        if "idx" in features[0]:
            batch["idx"] = [feature["idx"] for feature in features]

        batch_arrays = {}
        batch_arrays["input_ids"] = mx.array(batch["input_ids"], dtype=mx.int32)
        batch_arrays["attention_mask"] = mx.array(batch["attention_mask"], dtype=mx.int32)
        
        if "labels" in batch:
            batch_arrays["labels"] = mx.array(batch["labels"], dtype=mx.int32)
        
        if "position_ids" in batch:
            batch_arrays["position_ids"] = mx.array(batch["position_ids"], dtype=mx.int32)
        
        if "idx" in batch:
            batch_arrays["idx"] = batch["idx"]
        
        return batch_arrays


def get_question_latent_dataset(
    scheduled_stage: int,
    base_dataset: List[Dict],
    configs,
    start_id: int,
    latent_id: int,
    end_id: int,
    no_special_marker: bool = False,
) -> List[Dict]:
    def process_sample(sample):
        pad_latent_to_max = getattr(configs, 'pad_latent_to_max', False)
        max_latent_stage = getattr(configs, 'max_latent_stage', 0)
        c_thought = getattr(configs, 'c_thought', 1)
        
        if pad_latent_to_max:
            max_latent_stage = max_latent_stage
        else:
            max_latent_stage = min(max_latent_stage, len(sample["steps_tokenized"]))

        k = min(max_latent_stage, scheduled_stage)
        k *= c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * k
            + ([] if no_special_marker else [end_id])
        )

        return {
            "input_ids": tokens,
            "idx": sample["idx"],
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
        }

    processed_data = []
    for sample in base_dataset:
        processed_sample = process_sample(sample)
        processed_data.append(processed_sample)
    
    return processed_data


def get_cot_latent_dataset(
    scheduled_stage: int,
    base_dataset: List[Dict],
    configs,
    start_id: int,
    latent_id: int,
    end_id: int,
    no_special_marker: bool = False,
    shuffle: bool = False,
) -> List[Dict]:
    n_additional_tokens = 0 if no_special_marker else 2

    def process_sample(sample):
        uniform_prob = getattr(configs, 'uniform_prob', 0.0)
        max_latent_stage = getattr(configs, 'max_latent_stage', 0)
        pad_latent_to_max = getattr(configs, 'pad_latent_to_max', False)
        no_cot = getattr(configs, 'no_cot', False)
        c_thought = getattr(configs, 'c_thought', 1)
        
        if random.random() < uniform_prob:
            scheduled_stage_to_train = random.choice(list(range(len(sample["steps_tokenized"]) + 1)))
        else:
            scheduled_stage_to_train = scheduled_stage

        if scheduled_stage_to_train > max_latent_stage:
            n_skip_steps = 10000
            if pad_latent_to_max:
                n_latent_tokens = max_latent_stage
            else:
                n_latent_tokens = min(len(sample["steps_tokenized"]), max_latent_stage)
        else:
            n_skip_steps, n_latent_tokens = scheduled_stage_to_train, scheduled_stage_to_train

        if no_cot:
            n_skip_steps = 100
            n_latent_tokens = 0

        n_latent_tokens *= c_thought

        tokens = (
            sample["question_tokenized"]
            + ([] if no_special_marker else [start_id])
            + [latent_id] * n_latent_tokens
            + ([] if no_special_marker else [end_id])
            + list(itertools.chain.from_iterable(sample["steps_tokenized"][n_skip_steps:]))
            + sample["answer_tokenized"]
        )

        return {
            "input_ids": tokens,
            "labels": [-100] * (len(sample["question_tokenized"]) + n_latent_tokens + n_additional_tokens)
            + tokens[n_latent_tokens + n_additional_tokens + len(sample["question_tokenized"]):],
            "attention_mask": [1] * len(tokens),
            "idx": sample["idx"],
            "position_ids": list(range(len(tokens))),
        }

    processed_data = []
    for sample in base_dataset:
        processed_sample = process_sample(sample)
        processed_data.append(processed_sample)
    
    if shuffle:
        random.shuffle(processed_data)
    
    return processed_data
