import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map, tree_flatten, tree_unflatten
import json
import os
import yaml
import argparse
import gc
from tqdm import tqdm
import tiktoken

from gpt2 import GPT2Model, GPT2Config
from coconut import CoconutModel
from utils import Config, set_seed
from dataset import (
    get_dataset,
    get_question_latent_dataset, 
    get_cot_latent_dataset,
    Collator,
    DataLoader,
)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")

class GradientAccumulator:
    def __init__(self):
        self.accumulated_grads = None
        self.step_count = 0
    
    def accumulate(self, grads, accumulation_steps):
        scaled_grads = tree_map(lambda g: g / accumulation_steps, grads)
        
        if self.accumulated_grads is None:
            self.accumulated_grads = scaled_grads
        else:
            self.accumulated_grads = tree_map(lambda a, g: a + g, self.accumulated_grads, scaled_grads)
        
        self.step_count += 1
        return self.step_count
    
    def get_and_reset(self):
        grads = self.accumulated_grads
        self.accumulated_grads = None
        self.step_count = 0
        return grads
    
def setup_config_and_directories(config_file):
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)
    
    print("Config:", config_dict)
    configs = Config(config_dict)
    set_seed(configs.seed)
    
    save_dir = os.path.join(configs.save_path, configs.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cur_ckpts = os.listdir(save_dir)
    if len(cur_ckpts) > 0 and not configs.only_eval:
        print("Warning: found previous run, resuming from latest checkpoint")
        checkpoints = [f for f in cur_ckpts if f.startswith("checkpoint_")]
        checkpoints.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
        
        if checkpoints:
            latest_checkpoint = checkpoints[-1]
            epoch_num = int(latest_checkpoint.split("_")[1].split(".")[0])
            configs.resume = epoch_num
            configs.load_model_path = os.path.join(save_dir, latest_checkpoint)
            print(f"Loading from previous run epoch_{configs.resume}!")

    elif configs.resume != 0:
        if configs.load_model_path == "None":
            print(f"Warning: you want to skip the first {configs.resume} but you are not loading any existing checkpoint!")
        print(f"Loading from {configs.load_model_path} and skip the first {configs.resume} epochs")

    return configs, save_dir


def setup_tokenizer():
    base_tokenizer = tiktoken.get_encoding("gpt2")
    
    tokenizer = tiktoken.Encoding(
        name="gpt2_coconut",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={
            **base_tokenizer._special_tokens,
            "<|start-latent|>": 50257,
            "<|end-latent|>": 50258, 
            "<|latent|>": 50259,
        }
    )
    
    start_id = tokenizer.encode("<|start-latent|>", allowed_special={"<|start-latent|>"})[0]  
    end_id = tokenizer.encode("<|end-latent|>", allowed_special={"<|end-latent|>"})[0]
    latent_id = tokenizer.encode("<|latent|>", allowed_special={"<|latent|>"})[0]
    
    vocab_size = tokenizer.n_vocab
    print(f"Special token IDs: latent={latent_id}, start={start_id}, end={end_id}")
    print(f"Total vocab size: {vocab_size}")

    return tokenizer, base_tokenizer, start_id, end_id, latent_id, vocab_size

def load_weights(model, weights_path):
    print(f"Loading weights from {weights_path}")
    
    weights = mx.load(weights_path)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    
    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Loaded model with {nparams / 1e6:.3f} M parameters")
    return True


def setup_model(configs, vocab_size, base_tokenizer, start_id, end_id, latent_id):
    if hasattr(configs, 'model_config_path'):
        gpt_config = GPT2Config(config_path=configs.model_config_path)
        gpt_config.vocab_size = vocab_size
    else:
        gpt_config = GPT2Config(vocab_size=vocab_size)
    
    base_model = GPT2Model(gpt_config)
    
    loaded = False
    is_coconut_checkpoint = False
    if configs.load_model_path != "None":
        try:
            saved_weights = mx.load(configs.load_model_path)
            is_coconut_checkpoint = any(k.startswith("base_causallm") for k in saved_weights.keys())
            
            if configs.coconut and not is_coconut_checkpoint:
                loaded = load_weights(base_model, configs.load_model_path)
                print("Loaded base model weights into Coconut")
            elif not configs.coconut and is_coconut_checkpoint:
                print("Extracting base_causallm weights from coconut checkpoint for base model")
                base_causallm_weights = {}
                for key, value in saved_weights.items():
                    if key.startswith("base_causallm."):
                        new_key = key[len("base_causallm."):]
                        
                        if new_key.startswith("transformer."):
                            new_key = new_key[len("transformer."):]
                        
                        if new_key.startswith("lm_head."):
                            print(f"Skipping {key} - MLX GPT2Model uses tied embeddings")
                            continue
                        
                        base_causallm_weights[new_key] = value
                
                if base_causallm_weights:
                    base_model.update(tree_unflatten(list(base_causallm_weights.items())))
                    mx.eval(base_model.parameters())
                    print(f"Loaded {len(base_causallm_weights)} base_causallm parameters into base model")
                    loaded = True
                else:
                    print("Warning: No base_causallm weights found in checkpoint")
                    loaded = False
            elif configs.coconut and is_coconut_checkpoint:
                pass
            else:
                loaded = load_weights(base_model, configs.load_model_path)
                print("Loaded model weights")
        except Exception as e:
            print(f"Warning: Could not load weights from {configs.load_model_path}: {e}")

    should_init_tokens = (
        not (getattr(configs, 'cot', False) or getattr(configs, 'no_thoughts', False) or getattr(configs, 'no_cot', False))
        and not (configs.coconut and is_coconut_checkpoint)
    )
    
    if should_init_tokens:
        print(f"Initializing special tokens in the model")
        embed_layer = base_model.wte
        target_token = base_tokenizer.encode("<<")[0]
        
        for new_token_id in [latent_id, start_id, end_id]:
            if new_token_id >= base_tokenizer.n_vocab:
                embed_layer.weight[new_token_id] = embed_layer.weight[target_token]
                print(f"Initialized token {new_token_id} with embedding from token {target_token}")
    elif configs.coconut and is_coconut_checkpoint:
        print("Skipping special token initialization - coconut checkpoint has trained embeddings")

    if getattr(configs, 'no_thoughts', False):
        configs.c_thought = 0
        configs.coconut = False

    if configs.coconut:
        model = CoconutModel(
            base_causallm=base_model,
            latent_token_id=latent_id,
            start_latent_id=start_id, 
            end_latent_id=end_id,
            eos_token_id=base_tokenizer.eot_token
        )
        print(f"Created CoconutModel with latent_token_id={latent_id}, start_id={start_id}, end_id={end_id}")
        
        if configs.load_model_path != "None" and not loaded:
            saved_weights = mx.load(configs.load_model_path)
            
            base_causallm_weights = {}
            for key, value in saved_weights.items():
                if key.startswith("base_causallm."):
                    new_key = key[len("base_causallm."):]
                    
                    if new_key.startswith("transformer."):
                        new_key = new_key[len("transformer."):]
                    
                    if new_key.startswith("lm_head."):
                        print(f"Skipping {key} - MLX GPT2Model uses tied embeddings")
                        continue
                    
                    base_causallm_weights[new_key] = value
            
            if base_causallm_weights:
                model.base_causallm.update(tree_unflatten(list(base_causallm_weights.items())))
                mx.eval(model.base_causallm.parameters())
                print("Loaded Coconut model weights into base_causallm")
                
                try:
                    test_input = mx.array([[1, 2, 3]])
                    test_output = model.base_causallm(input_ids=test_input)
                    print(f"✓ base_causallm test successful, output shape: {test_output.logits.shape}")
                except Exception as e:
                    print(f"✗ base_causallm test failed: {e}")
            else:
                print("Warning: No base_causallm weights found in checkpoint")
                
            other_weights = {k: v for k, v in saved_weights.items() if not k.startswith("base_causallm.")}
            if other_weights:
                model.update(tree_unflatten(list(other_weights.items())))
                mx.eval(model.parameters())
                print(f"Loaded {len(other_weights)} additional parameters")
    else:
        model = base_model

    nparams = sum(x.size for k, x in tree_flatten(model.parameters()))
    print(f"Model parameters: {nparams / 1e6:.3f} M parameters")

    return model


def setup_data(configs, tokenizer):
    val_data = json.load(open(configs.val_path))
    question_val = [d["question"] for d in val_data]
    answers_val = [d["answer"].replace(",", "").strip() for d in val_data]
    cot_val = ["\n".join(d["steps"]) for d in val_data]

    base_dataset_valid = get_dataset(
        configs.val_path, tokenizer, max_size=32 if configs.debug else 100000000
    )

    base_dataset_train = None
    if not configs.only_eval:
        base_dataset_train = get_dataset(
            configs.train_path, tokenizer, max_size=5000 if configs.debug else 100000000
        )

    return question_val, answers_val, cot_val, base_dataset_valid, base_dataset_train


def decode_and_extract_answer(tokenizer, token_ids):
    if hasattr(token_ids, 'tolist'):
        token_ids = token_ids.tolist()
    
    token_ids = [t for t in token_ids if t < tokenizer.n_vocab]
    text_output = tokenizer.decode(token_ids)
    
    answer_output = text_output.split("#")[-1].replace(",", "").strip()
    cot_output = ("\n".join(text_output.split("\n")[1:])).split("#")[0].strip()
    
    return answer_output, cot_output, text_output


def generate(model, input_ids, max_new_tokens, eos_token_id, temperature=0.0):
    try:
        original_input = input_ids[0].tolist() if hasattr(input_ids[0], 'tolist') else input_ids[0]
        input_length = len(original_input)
        generated = original_input.copy()
        
        for _ in range(max_new_tokens):
            current_input = mx.array([generated])
            
            if hasattr(model, 'base_causallm'):
                attention_mask = mx.ones((1, len(generated)))
                position_ids = mx.arange(len(generated)).reshape(1, -1)
                
                outputs = model(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=None
                )
            else:
                outputs = model(input_ids=current_input)
            
            if outputs is None:
                break
            
            if hasattr(outputs, 'logits'):
                if outputs.logits is None:
                    break
                next_token_logits = outputs.logits[0, -1, :]
            else:
                if outputs is None or (hasattr(outputs, '__len__') and len(outputs) == 0):
                    break
                if outputs[0] is None:
                    break
                next_token_logits = outputs[0, -1, :]
            
            if next_token_logits is None:
                break

            if temperature > 0:
                next_token_logits = next_token_logits / temperature
                probs = mx.softmax(next_token_logits)
                next_token = mx.random.categorical(mx.log(probs)).item()
            else:
                next_token = mx.argmax(next_token_logits).item()
            
            if next_token == eos_token_id:
                break
            generated.append(next_token)
        
        new_tokens = generated[input_length:]
        return mx.array([new_tokens])
        
    except Exception as e:
        print(f"Error in generate function: {e}")
        return mx.array([[]])


def train_epoch(model, dataloader, optimizer, configs, grad_accumulator, wandb_run, epoch, total_train_steps):
    model.train()
    train_loss = 0
    
    total_optimization_steps = len(dataloader) // configs.gradient_accumulation_steps
    if len(dataloader) % configs.gradient_accumulation_steps != 0:
        total_optimization_steps += 1
        
    pbar = tqdm(
        desc=f"Training Epoch: {epoch+1}/{configs.num_epochs}",
        total=total_optimization_steps,
        colour="blue",
        dynamic_ncols=True
    )

    actual_step = 0
    
    for step, batch in enumerate(dataloader):
        if step == 0 and wandb_run:
            print("Logging training data sample")
            cur_bs = min(len(batch["input_ids"]), 2)
            for data_idx in range(cur_bs):
                text_str = ""
                for token_idx in range(len(batch["input_ids"][data_idx])):
                    text_str += (
                        f"{batch['input_ids'][data_idx][token_idx].item()} "
                        f"{batch['labels'][data_idx][token_idx].item()} "
                        f"\n"
                    )
        total_train_steps += 1

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        position_ids = batch["position_ids"] if "position_ids" in batch else None

        def loss_fn(params):
            model.update(params)
            
            if configs.coconut:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    position_ids=position_ids
                )
                return outputs.loss.mean()
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids
                )
                logits = outputs.logits
                
                shift_logits = logits[..., :-1, :]
                shift_labels = labels[..., 1:]
                
                valid_mask = shift_labels != -100
                shift_labels = mx.where(valid_mask, shift_labels, 0)
                
                loss = nn.losses.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.shape[-1]),
                    shift_labels.reshape(-1),
                    reduction='none'
                )
                loss = mx.where(valid_mask.reshape(-1), loss, 0)
                loss = loss.sum() / valid_mask.sum()
                
                return loss

        loss, grads = mx.value_and_grad(loss_fn)(model.parameters())
        step_count = grad_accumulator.accumulate(grads, configs.gradient_accumulation_steps)

        if step_count >= configs.gradient_accumulation_steps or step == len(dataloader) - 1:
            accumulated_grads = grad_accumulator.get_and_reset()
            optimizer.update(model, accumulated_grads)
            mx.eval(model.parameters())
            
            actual_step += 1
            pbar.update(1)
            
            avg_loss_for_step = loss.item()
            train_loss += avg_loss_for_step
            pbar.set_postfix(loss=f"{avg_loss_for_step:.4f}")
        
        if wandb_run:
            log_dict = {
                "train/epoch": epoch + 1,
                "train/step": total_train_steps,
                "train/loss": loss.item(),
                "train/global_step": step + epoch * len(dataloader),
            }
            wandb_run.log(log_dict)
    
    pbar.close()
    avg_train_loss = train_loss / actual_step if actual_step > 0 else 0
    return avg_train_loss, total_train_steps


def evaluate_loss(model, dataloader, configs):
    print("Computing validation loss...")
    model.eval()
    val_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation Loss", colour="green"):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        position_ids = batch["position_ids"] if "position_ids" in batch else None

        if configs.coconut:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids
            )
            val_loss += outputs.loss.mean().item()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            )
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            
            valid_mask = shift_labels != -100
            shift_labels = mx.where(valid_mask, shift_labels, 0)
            
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]),
                shift_labels.reshape(-1),
                reduction='none'
            )
            loss = mx.where(valid_mask.reshape(-1), loss, 0)
            loss = loss.sum() / valid_mask.sum()
            
            val_loss += loss.item()
        
        num_batches += 1

    avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
    print(f"Validation loss: {avg_val_loss:.4f}")
    return avg_val_loss


def evaluate_generation(model, dataset_gen_val, collator, question_val, answers_val, cot_val, tokenizer, max_new_tokens):
    print("Evaluating generation accuracy...")
    model.eval()
    mx.eval(model.parameters())
    
    correct = 0
    correct_cot = 0
    total = 0
    
    pbar = tqdm(range(len(dataset_gen_val)), desc="Test Accuracy", colour="blue", dynamic_ncols=True)
    
    for idx in pbar:
        data_point = dataset_gen_val[idx]
        batch = collator([data_point])
        
        test_idx = data_point.get('idx', idx)
        answer = answers_val[test_idx]
        answer_cot = cot_val[test_idx] 
        
        total += 1
        input_ids = batch["input_ids"]

        try:
            if hasattr(model, 'generate') and hasattr(model, 'base_causallm'):
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=batch["attention_mask"] if "attention_mask" in batch else mx.ones_like(input_ids),
                    max_new_tokens=max_new_tokens,
                    synced_gpus=False
                )
                input_length = input_ids.shape[1]
                new_tokens = generated_ids[0][input_length:].tolist()
            elif hasattr(model, 'generate') and hasattr(model, 'wte'):
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    eos_token_id=tokenizer.eot_token
                )
                input_length = input_ids.shape[1]
                new_tokens = generated_ids[0][input_length:].tolist()
            else:
                generated_ids = generate(model, input_ids, max_new_tokens, tokenizer.eot_token)
                new_tokens = generated_ids[0].tolist()
            
            answer_output, cot_output, text_output = decode_and_extract_answer(tokenizer, mx.array(new_tokens))
            
            if answer_output == answer:
                correct += 1
            if cot_output == answer_cot:
                correct_cot += 1
                
        except Exception as e:
            print(f"Generation error at idx {idx}: {e}")
            answer_output, cot_output, text_output = "", "", ""
            
        if idx < 5:
            print(f"\nQuestion {test_idx}: Expected='{answer}', Generated='{answer_output}'")
            if idx < 2 and text_output:
                print(f"Generated text: '{text_output[:200]}...'")
                if "#" in text_output:
                    print(f"Contains {text_output.count('#')} hash characters")

        accuracy = correct / total if total > 0 else 0
        pbar.set_description(f"Test accuracy: {accuracy:.2%}")

    pbar.close()
    
    accuracy = correct / total if total > 0 else 0
    cot_accuracy = correct_cot / total if total > 0 else 0
    
    print(f"Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"CoT accuracy: {correct_cot}/{total} = {cot_accuracy:.4f}")

    return accuracy, cot_accuracy


def main():
    parser = argparse.ArgumentParser(description="Coconut MLX Training")
    parser.add_argument("config_file", help="Path to config YAML file")
    args = parser.parse_args()

    configs, save_dir = setup_config_and_directories(args.config_file)
    tokenizer, base_tokenizer, start_id, end_id, latent_id, vocab_size = setup_tokenizer()
    model = setup_model(configs, vocab_size, base_tokenizer, start_id, end_id, latent_id)
    question_val, answers_val, cot_val, base_dataset_valid, base_dataset_train = setup_data(configs, tokenizer)
    
    max_new_tokens = 64 if "gsm" in configs.val_path else 128

    optimizer = None
    if not getattr(configs, 'reset_optimizer', False):
        optimizer = optim.AdamW(learning_rate=configs.lr, weight_decay=configs.weight_decay)

    wandb_run = None
    if WANDB_AVAILABLE and not configs.debug and not configs.only_eval:
        wandb_run = wandb.init(
            project=getattr(configs, 'project', 'coconut-mlx'),
            name=configs.name,
            config=dict(configs.__dict__)
        )

    best_acc = 0
    total_train_steps = 0
    grad_accumulator = GradientAccumulator()
    collator = Collator(tokenizer, latent_id=latent_id, label_pad_token_id=-100)

    for epoch in range(configs.resume, configs.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{configs.num_epochs}")
        print(f"{'='*60}")
        
        if configs.cot or getattr(configs, 'no_cot', False):
            scheduled_stage = 0
        else:
            scheduled_stage = epoch // configs.epochs_per_stage
        
        print(f"Training stage: {scheduled_stage}")

        dataset_gen_val = get_question_latent_dataset(
            scheduled_stage,
            base_dataset_valid,
            configs,
            start_id,
            latent_id,
            end_id,
            no_special_marker=configs.cot or getattr(configs, 'no_cot', False) or getattr(configs, 'no_thoughts', False),
        )

        if not configs.only_eval:
            dataset_train = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_train,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or getattr(configs, 'no_cot', False) or getattr(configs, 'no_thoughts', False),
                shuffle=True,
            )

            if getattr(configs, 'reset_optimizer', False):
                del optimizer
                optimizer = optim.AdamW(learning_rate=configs.lr, weight_decay=configs.weight_decay)
                print("Reset optimizer")

            train_dataloader = DataLoader(dataset_train, configs.batch_size_training, collator, shuffle=True)
            
            avg_train_loss, total_train_steps = train_epoch(
                model, train_dataloader, optimizer, configs, grad_accumulator, 
                wandb_run, epoch, total_train_steps
            )
            
            print(f"Average training loss: {avg_train_loss:.4f}")
            
            if not getattr(configs, 'save_only_improve', False) and not configs.debug:
                checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}")
                params_dict = dict(tree_flatten(model.parameters()))
                mx.save_safetensors(checkpoint_path + ".safetensors", params_dict)
                print(f"Saved checkpoint to {checkpoint_path}.safetensors")

            dataset_loss_val = get_cot_latent_dataset(
                scheduled_stage,
                base_dataset_valid,
                configs,
                start_id,
                latent_id,
                end_id,
                no_special_marker=configs.cot or getattr(configs, 'no_cot', False) or getattr(configs, 'no_thoughts', False),
            )
            
            val_dataloader = DataLoader(dataset_loss_val, configs.batch_size_training, collator, shuffle=False)
            avg_val_loss = evaluate_loss(model, val_dataloader, configs)
            
            if wandb_run:
                wandb_run.log({
                    "train/epoch": epoch + 1,
                    "train/avg_loss": avg_train_loss,
                    "eval/loss": avg_val_loss
                })

        accuracy, cot_accuracy = evaluate_generation(
            model, dataset_gen_val, collator, question_val, answers_val, cot_val, 
            tokenizer, max_new_tokens
        )

        if wandb_run:
            wandb_run.log({"eval/acc": accuracy, "eval/cot_em": cot_accuracy})

        if configs.only_eval:
            print("Evaluation complete. Exiting...")
            break

        if accuracy > best_acc and getattr(configs, 'save_only_improve', False) and not configs.debug:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_{epoch + 1}")
            params_dict = dict(tree_flatten(model.parameters()))
            mx.save_safetensors(checkpoint_path + ".safetensors", params_dict)
            print(f"New best accuracy {accuracy:.4f}, saved checkpoint to {checkpoint_path}.safetensors")
            best_acc = accuracy

        gc.collect()
        mx.metal.clear_cache()

    if wandb_run:
        wandb_run.finish()

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()