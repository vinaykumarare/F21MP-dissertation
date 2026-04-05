# Evaluating Instruction vs Question-Based Safety Training for Large Language Models: A Study on Safety Effectiveness and Cross-Format Robustness

This repository contains the code, datasets, and experimental outputs for my MSc dissertation. The project investigates whether the format of safety training data affects the safety behaviour of language models after fine-tuning.

Two safety training formats are compared: instruction-based prompts and question-based prompts. The models are trained using general instruction data from Alpaca and safety data from SafeText. They are then evaluated on unseen SafeText prompts, AdvBench, and SORRY-Bench to assess in-domain refusal behaviour, cross-format robustness, and out-of-domain safety generalisation.
