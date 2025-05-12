# Fast High-dimensional Approximate Nearest Neighbor Search with Efficient Index Time and Space

## Introduction

This is the official implementation of the paper [Fast High-dimensional Approximate Nearest Neighbor Search with Efficient Index Time and Space](https://arxiv.org/abs/2411.06158).

MRQ leverage leverages data distribution to achieve better distance correction and a higher vector compression ratio. It significantly outperforms state-of-the-art AKNN search methods based on graph or vector quantization, achieving up to a 3x efficiency speed-up with only 1/3 length of quantized code while maintaining the same accuracy.

## Requirements

* C++17
* Python
* OpenMP

## Directory Structure

```
.
├── include            # Helper functions for HNSW, IVF, and FastScan
├── script             # Python file for data preprocessing
├── src                # Code for running our method
├── test               # Scripts for reproduction
│   ├── GraphBitQ
│   ├── HNSW
│   ├── RaBitQ
│   └── RESQ
├── mkdir.sh           # Creates project directories, builds with CMake
├── run.sh             # Specify the tests to be run in ./test
└── set.sh             # Specify datasets and paths
```

## Reproduction

### Prepare Datasets

All datasets we used for the evaluation can be downloaded from [ANN-Benchmark](https://github.com/erikbern/ann-benchmarks) or other public repo. Download the base and query set to the `./DATA` directory and ensure they are in `.fvecs`/`.ivecs` format.

### Runing Tests

Step 1. Specify datasets in `set.sh`

```zsh
export datasets=("gist" "msong" "deep1M" "OpenAI-1536" "OpenAI-3072" "msmarc-small")
export store_path=./DATA
export result_path=./results
```
Step 2. Set indexing and searching parameters (Optional)

To run tests on new datasets, you must set parameters for them.

Indexing parameters:

- RabitQ: 
    - `B`: in `./test/RaBitQ/test_index.sh`, number of bits used in quantization. Determined by dimension `D` with `B = (D + 63) // 64 * 64`
    - `BB`: in `./src/RaBitQ/index.cpp`, same as `B`
    - `DIM`: in `./src/RaBitQ/index.cpp`, dimension of the dataset.
  
- MRQ/MRQ+: 
    - `B`: in `./test/RESQ/test_index.sh`, the number of dimensions after PCA and the number of bits used in quantization. Should be a multiple of 64.
    - `BB` and `DIM`: in `./src/ResBitQ/res_index.cpp`.

Searching parameters:

- RabitQ: 
    - `B`/`BB`: same to those used in indexing.
    - `probe_base`: in `./src/RaBitQ/search.cpp`, search with `nprobe` in range(probe_base, 20*probe_base, probe_base).
  
- MRQ/MRQ+: 
    - `B`/`BB`: same to those used in indexing.
    - `probe_base`: Search with `nprobe` in range(probe_base, 20*probe_base, probe_base).
    - `var_count`: The factor for residual error bound. Balancing accuracy and efficiency.

- HNSW:
    - `efSearch`: in `./test/HNSW/test_naive_search.sh`


Step 3. Choose which test you wish to run in `run.sh`

Tests include:

* RabitQ: `./rabit_index` for indexing; `./rabit_scan_search` and `./rabit_fast_scan_search` for searching.
* MRQ: `./res_index` for indexing; `./res_search` for searching.
* MRQ+: `./res_split_index` for indexing; `./res_split_search` for searching.
* HNSW: `./index_hnsw` for indexing; `./search_hnsw` for searching.

  


