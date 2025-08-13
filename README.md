# MLX Coconut

This codebase is an MLX port of the [official implementation](https://github.com/facebookresearch/coconut) of [Training Large Language Models to Reason in a Continuous Latent Space](https://arxiv.org/abs/2412.06769). This README is based heavily on the official repository's [README](https://github.com/facebookresearch/coconut/blob/main/README.md).

## Getting Started

Clone the repo:

```
git clone git@github.com:vincentamato/mlx-coconut.git
cd mlx-coconut
```

Setup the environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The data for training and evaluation should be presented as a json file like below:

```python
[
  {
    "question": "...",
    "answer": "...",
    "steps": ["...", "...", ...]
  },
  ...
]
```

The file should contain a list of data points. Each data point is composed of a question (str), an answer (str), and a list of steps (str), where each of them is a string.

For example, you can download and process the [GSM8K](https://arxiv.org/abs/2110.14168) dataset (with [augmented training and validation sets](https://github.com/da03/Internalize_CoT_Step_by_Step/tree/e06a32ee5e4cd117171daeb4755d2a97ece62761/data/gsm8k)) by running:

```bash
bash preprocessing/gsm_icot.bash
```

## Arguments

The configuration of a run should be specified in a yaml file (an example can be found [here](args/gsm_coconut.yaml)).

- **General settings**

  - **project**: Project name for wandb
  - **save_path**: Your path to store the checkpoints
  - **only_eval**: If true, only load a model and test on the data from `val_path` (must used along with `load_model_path`). Otherwise, train the model on `train_path` and test on `val_path` after every epoch.

- **Method**

  - **coconut**: Train coconut model
  - **cot**: Train cot model
  - **no_thoughts**: Train coconut (w/o thought) model
  - **no_cot**: Train no-cot model

- **Training settings**

  - **c_thought**: Number of continuous thoughts for each reasoning step
  - **epochs_per_stage**: Number of epochs for every training stage
  - **max_latent_stage**: The maximum number of training stages (in addition to the initial stage)
  - **pad_latent_to_max**: If the number of reasoning steps is fewer than the index of current training stage, pad the number of continuous thoughts.
  - **save_only_improve**: Save the model only when there the best validation accuracy is updated. Recommended to set `False` for Coconut model training, because otherwise the checkpoints in the last stage might now get saved.
  - **uniform_prob**: The probability to mix data from other stages. 0 for standard experiment, 0.3 for analysis experiment.
  - **load_model_path**: The path to a checkpoint to load. Used in three cases: (1) for evaluation (2) to train a CoT-tuned model from base GPT-2 (3) to initialize coconut from a CoT-tuned model.
  - **model_config_path**: The path to GPT-2's configuration file.
  - **seed**: Random seed.
  - **resume**: The epoch to resume. Can be used when we want to skip the initial training stages.
  - **train_path**: Path to the training set.
  - **val_path**: Path to the validation or test set (depending on `only_eval`)
  - **reset_optimizer**: Whether to reset the optimizer when swtiching training stages.
  - **batch_size_training**: Batch size to train the model per GPU.
  - **debug**: If true, there is no wandb and model saving. A subset of data will be used.
  - **gradient_accumulation_steps**: Gradient accumulation steps
  - **num_epochs**: Maximum training epoches.
  - **lr**: Learning rate
  - **weight_decay**: Weight decay

## Training

Run the following command:

```bash
python run.py PATH_TO_ARGS
```

## Reproducing Experiments

Here I provide instructions to reproduce the experiments in the paper. Due to training dynamics with MLX and Apple Silicon, results may not be exact to the paper. The MLX-friendly base GPT-2 model and configuration file can be found [here](https://huggingface.co/vincentamato/mlx-coconut/tree/maint). For results that closely align with those reported in the paper, a high-performance Apple Silicon chip is recommended.

### GSM8K

Preprocessing data:

```bash
bash preprocessing/gsm_icot.bash
```

First train the model with CoT (as the stage 0 training)

```bash
python run.py args/gsm_cot.yaml
```

Select a checkpoint as the initialization of Coconut (the validation accuracy is expected to be around 40%). Replace the `load_model_path` in the [args/gsm_coconut.yaml](args/gsm_coconut.yaml) with your selected checkpoint, and run:

```bash
python run.py args/gsm_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/gsm_coconut_eval.yaml](args/gsm_coconut_eval.yaml). To evaluate:

```bash
python run.py args/gsm_coconut_eval.yaml
```

### ProntoQA

Please clone the official [github repo](https://github.com/asaparov/prontoqa/tree/f0145b867b3c106285ec9ea1941a3f6eb7c6162d) of [ProntoQA](https://arxiv.org/pdf/2210.01240) and generate a raw dataset with:

```bash
cd prontoqa
python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1
```

Then copy the generated `5hop_0shot_random.json` file to `data` directory, and preprocess the dataset with:

```bash
python preprocessing/prontoqa.py
```

Then run the following to train the model:

```bash
python run.py args/prontoqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
python run.py args/prosqa_coconut_eval.yaml
```

### ProsQA

The ProsQA dataset is at [data/prosqa\_\*.json](data).

Then run the following to train the model:

```bash
python run.py args/prosqa_coconut.yaml
```

Find the checkpoint with best validation accuracy, and put the path as `load_model_path` in [args/prosqa_coconut_eval.yaml](args/prosqa_coconut_eval.yaml). To evaluate:

```bash
python run.py args/prosqa_coconut_eval.yaml
```

## Pretrained Model Evaluation on ProsQA

I trained Coconut on the ProsQA dataset using the official codebase on 4 A100 GPUs. I then evaluated them on my M4 Pro MacBook Pro. Both models can be downloaded [here](https://huggingface.co/vincentamato/mlx-coconut/tree/main).

### Evaluating the CoT-tuned Model

Point the `load_model_path` to the downloaded CoT-tuned model and run:

```bash
python run.py args/prosqa_cot_eval.yaml
```

You should receive around 56% accuracy.

### Evaluating the Coconut Model

Point the `load_model_path` to the downloaded Coconut model and run:

```bash
python run.py args/prosqa_coconut_eval.yaml
```

You should receive around 79% accuracy.

## Citations

```bibtex
@article{hao2024training,
  title={Training Large Language Models to Reason in a Continuous Latent Space},
  author={Hao, Shibo and Sukhbaatar, Sainbayar and Su, DiJia and Li, Xian and Hu, Zhiting and Weston, Jason and Tian, Yuandong},
  journal={arXiv preprint arXiv:2412.06769},
  year={2024}
}
```
