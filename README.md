# Haplograph → Evo 2

Genome Cluster Engine Project

Established and optimized the haploblocks method for discretizing genomic data (https://osf.io/xhkc3)
Generated binary strings from haploblock clusters and did some basic linear modeling. Leveraged XGBoost and deep learning. Results were promising but not conclusive.
Processed the 1000 genomes data set with the haploblocks method (paper submitted)
Established and optimized the haplographs method (manuscript in preparation)

Here’s what we are currently working on.

Establishing what use cases respond better to a simpler clustering method (linclust)
Establishing what use cases respond best to:
Associated binary strings of haploblocks going straight into a DNA foundation model
Using XGBoost to come up with cogent groups of haploblocks during analysis
Building a haplograph and using node2vec to process the phenotype-genotype relationships
Building a haplograph and using a GNN (PyTorch) to isolate subnetworks and create contextualized vectors
Using XGBoost to come up with cogent groups of haploblocks during analysis of haplograph data
Testing all of the above conditions in both cancer and germline disease
Testing in what conditions early fusion or late fusion of proteomics data with genomic data and phenotype but is more effective


## Testing pertubations of the Haplograph --> Evo2 genomic model to see what works best in particular settings.  

As a POC, builds a haploblock co-occurrence graph per chromosome from the [1000 Genomes haplograph](https://data.haploblocks.org/haplograph/1000G/), turns its nodes into embeddings with a GNN or XGBoost, and fine-tunes Evo 2 (7B) on those embeddings

For more see: [haploblocks.org](haploblocks.org)

## Dependencies

- Python 3.11 or 3.12, Linux, NVIDIA GPU with CUDA 12.1+ (80 GB recommended for step 4)
- numpy, pandas, scipy (all steps)
- torch (node2vec and GNN embeddings, Evo 2)
- xgboost (XGBoost embeddings)
- evo2 + flash-attn (step 4)

## Install on Nvidia Brev

```bash
# 1. Install the CUDA Toolkit and cuDNN (one time, system-wide)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# CUDA 12.8 toolkit (includes nvcc) + cuDNN 9 headers for Transformer Engine
sudo apt install -y cuda-toolkit-12-8 libcudnn9-dev-cuda-12 build-essential

# 2. Set CUDA_HOME
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version   # should report 12.8

# 3. Create a virtual environment and install the Python packages
python3.12 -m venv ~/evo2-env
source ~/evo2-env/bin/activate
pip install --upgrade pip setuptools wheel ninja packaging

# PyTorch built for CUDA 12.8 (matches the toolkit above)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# Transformer Engine (needed for the 1B, 20B and 40B models; skip for 7B only)
sudo apt install -y libnccl2 libnccl-dev
pip install --no-build-isolation "transformer_engine[pytorch]"

# Flash Attention, then Evo 2
MAX_JOBS=4 pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
export NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE=1

# Other packages
pip install numpy pandas scipy xgboost
```


## Usage

Run the scripts in order. Step 1 writes the graph to `haplograph/genome/`; steps 2–4 read and write there (change with `--graph-dir`).

```bash
python 1_build_graph.py                            # whole genome (or --chroms chr21,chr22)
python 2_ancestry_labels.py                        # needed for xgboost or --objective ancestry
python 3_node_embeddings.py --method node2vec      # or gnn / xgboost
python 4_finetune_evo2.py   --method node2vec      # --objective next (default) or ancestry
```

Add `--chroms chr21,chr22` to any step for a quick test run, and `--help` to see all options.
