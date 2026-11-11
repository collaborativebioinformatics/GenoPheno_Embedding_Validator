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

`install_evo2.sh` sets up everything on an RTX PRO 6000 Blackwell or an L40S instance: the CUDA 12.8 toolkit, a Python 3.12 virtual environment at ~/evo2-env, PyTorch 2.7.1, flash-attn 2.8.0.post2, the Evo 2 light install (7B models, no Transformer Engine) and the other packages, then runs a test generation. It detects the GPU automatically; the only difference is that on the RTX PRO 6000 (sm_120) flash-attn is always compiled from source, which takes 20–40 minutes, while on the L40S (sm_89) a prebuilt wheel is usually used and the install finishes in a few minutes. The RTX PRO 6000's 96 GB is recommended for fine-tuning in step 4, since the L40S's 48 GB limits sequence length and how many blocks can be unfrozen. If the build gets killed for lack of RAM, lower MAX_JOBS.

```bash
bash install_evo2.sh          # auto-detect GPU (or: bash install_evo2.sh rtx / l40s)
source ~/evo2-env/bin/activate
```

## Usage

Run the scripts in order. Step 1 writes the graph to `haplograph/genome/`; steps 2–4 read and write there (change with `--graph-dir`).

```bash
python3 1_build_graph.py                            # whole genome (or --chroms chr21,chr22)
python3 2_ancestry_labels.py                        # needed for xgboost or --objective ancestry
python3 3_node_embeddings.py --method node2vec      # or gnn / xgboost
python3 4_finetune_evo2.py   --method node2vec      # --objective next (default) or ancestry
```

Add `--chroms chr21,chr22` to any step for a quick test run, and `--help` to see all options.

### Example test run

This runs the full pipeline as a quick test on chr 21. The first command builds the haploblock co-occurrence graph for chr21 and writes it to `haplograph/chr21/`, and the second assigns 1000 Genomes ancestry labels to that graph. The third turns the graph nodes into embeddings with node2vec. The fourth fine-tunes Evo2 for one epoch to predict ancestry from those embeddings, with `--train-blocks 0` keeping all Evo2 blocks frozen so only the added layers are trained.

```bash
python3 1_build_graph.py --chroms chr21
python3 2_ancestry_labels.py --graph-dir haplograph/chr21/
python3 3_node_embeddings.py --method node2vec --graph-dir haplograph/chr21/
python3 4_finetune_evo2.py   --method node2vec --graph-dir haplograph/chr21/ --objective ancestry --train-blocks 0 --epochs 1
```

### Running the whole genome 

The example above runs on chr21 as a quick test. To run the whole genome, build the graph across all autosomes (1_build_graph.py with no --chroms covers chr1–chr22) and point the later steps at haplograph/genome/.

I ran this on an RTX PRO 6000 Blackwell for the hackathon. Worth flagging up front: the bottleneck here is host RAM, not GPU. Steps 1–3 are CPU/memory-bound. The step that strains is 3_node_embeddings.py --method xgboost: its leaf-embedding stage one-hot encodes every node’s leaf index across all trees, and at genome scale (11M nodes) that transient matrix is large enough to exhaust ~250 GB at the default --xgb-trees 100 --xgb-depth 6. On a 141 GB machine this gets OOM-killed right after the accuracy line, with no traceback.

The fix is to shrink the leaf matrix with fewer, shallower trees. --xgb-trees 20 --xgb-depth 3 keeps peak RAM around 120 GB and still reaches ~0.98 ancestry accuracy (versus ~0.99 at default), since the super-population signal is easily separable. --device cpu avoids a GPU/CPU device-mismatch in XGBoost’s apply step.

'''bash
python3 1_build_graph.py
python3 2_ancestry_labels.py --graph-dir haplograph/genome/
python3 3_node_embeddings.py --method xgboost --graph-dir haplograph/genome/ --device cpu --xgb-trees 20 --xgb-depth 3
python3 4_finetune_evo2.py   --method xgboost --graph-dir haplograph/genome/ --objective ancestry --train-blocks 0 --epochs 1
'''
