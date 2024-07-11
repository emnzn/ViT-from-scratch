# Vision Transformer From Scratch

![Vision Transformer Architecture](assets/figures/ViT-Architecture.png)

A PyTorch implementation of the [Vision Transformer](https://arxiv.org/pdf/2010.11929) (**An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**)

## Objective
To create the Vision Transformer from scratch and validate its performance against the official PyTorch implementation.

## Repository Contents

This repository contains constructors for:

| Model      | Layers | Hidden size D | MLP size | Heads | Params |
|------------|--------|---------------|----------|-------|--------|
| ViT-Base   | 12     | 768           | 3072     | 12    | 86M    |
| ViT-Large  | 24     | 1024          | 4096     | 16    | 307M   |
| ViT-Huge   | 32     | 1280          | 5120     | 16    | 632M   |

## Usage
