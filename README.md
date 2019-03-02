# PyTorch Position Embedding

[![Travis](https://travis-ci.org/CyberZHG/torch-position-embedding.svg)](https://travis-ci.org/CyberZHG/torch-position-embedding)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/torch-position-embedding/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/torch-position-embedding)

## Install

```bash
pip install torch-position-embedding
```

## Usage

```python
from torch_position-embedding import PositionEmbedding

PositionEmbedding(num_embeddings=5, embedding_dim=10, mode=PositionEmbedding.MODE_ADD)
```

Modes:

* `MODE_EXPAND`: negative indices could be used to represent relative positions.
* `MODE_ADD`: add position embedding to the original tensor.
* `MODE_CAT`: concatenate position embedding to the original tensor.
