# Evaluating Instruction vs Question-Based Safety Training for Large Language Models: A Study on Safety Effectiveness and Cross-Format Robustness

This dissertation explores whether the format of safety training data affects how safely a large language model behaves after fine-tuning. A base model (Mistral-7B) was fine-tuned into two versions, one trained on instruction-style safety prompts and one trained on question-style safety prompts, with everything else kept identical. The models were then tested to see how well they refused unsafe requests, whether that refusal held up when the prompt format changed, and whether safety generalised to harm categories outside the training data. This repository contains the datasets, training scripts, model responses, and results from all experiments.

# Language Model

The base model used in this dissertation is Mistral-7B-v0.3, a decoder-only transformer model with 7.3 billion parameters. The base version was chosen deliberately — it had not been instruction-tuned or safety-trained before, which means any safety behaviour observed after fine-tuning can be attributed entirely to the training done in this experiment.
The model was downloaded from [HuggingFace]: (https://huggingface.co/mistralai/Mistral-7B-v0.3).
