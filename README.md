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
python 4a_finetune_evo2_embeddings.py --method node2vec   # --objective next (default) or ancestry
```

Add `--chroms chr21,chr22` to any step for a quick test run, and `--help` to see all options.

### Step 4a with NVIDIA BioNeMo (experimental)

`4a_finetune_evo2_embeddings_bionemo.py` does the same fine-tuning as step 4a, but loads and runs Evo 2 through NVIDIA BioNeMo (NeMo 2 / Megatron Core) instead of the evo2 package. It has not been tested on a GPU yet and may need adjusting for your BioNeMo version (the affected parts are marked VERSION-SENSITIVE in the code).

It runs inside the BioNeMo Framework container (nvcr.io/nvidia/clara/bionemo-framework) and needs an Evo 2 checkpoint in NeMo2 format:

```bash
evo2_convert_to_nemo2 --model-path hf://arcinstitute/savanna_evo2_7b_base --model-size 7b --output-dir nemo2_evo2_7b_8k
python 4a_finetune_evo2_embeddings_bionemo.py --method gnn --ckpt-dir nemo2_evo2_7b_8k
```

To fine-tune on hash-phenotype associations instead, use step 4b on its own (no steps 1–3 needed). Its input is a TSV with two columns, a genomic hash (0/1 string) and a phenotype (e.g. height).

```bash
python 4b_finetune_evo2_hash.py --hash-pheno hash_height.tsv   # numeric -> regression, labels -> classification
```

### Example test run

This runs the full pipeline as a quick test on chr 21. The first command builds the haploblock co-occurrence graph for chr21 and writes it to `haplograph/chr21/`, and the second assigns 1000 Genomes ancestry labels to that graph. The third turns the graph nodes into embeddings with node2vec. The fourth fine-tunes Evo2 for one epoch to predict ancestry from those embeddings, with `--train-blocks 0` keeping all Evo2 blocks frozen so only the added layers are trained.

```bash
python3 1_build_graph.py --chroms chr21
python3 2_ancestry_labels.py --graph-dir haplograph/chr21/
python3 3_node_embeddings.py --method node2vec --graph-dir haplograph/chr21/
python3 4_finetune_evo2.py   --method node2vec --graph-dir haplograph/chr21/ --objective ancestry --train-blocks 0 --epochs 1
```

### Validation

`5_validate_evo2.py` checks one fine-tuned model (a .pt file) against a test set. The test set is a text file with one individual per line (--test-individuals). By default it is the set the model held out during training, and the train set is every other individual in the graph.

```bash
python 5_validate_evo2.py --model haplograph/genome/evo2_next_gnn.pt --tasks validation_tasks.txt
python 5_validate_evo2.py --model evo2_next_gnn.pt --test-individuals test.txt --gwas height_hg38.tsv
```

The checks to run are listed in a tasks file (`validation_tasks.txt` is an example), one per line with key=value parameters. Add a line to run a check again with other settings. `--list-tasks` shows every task and its parameters.

- model_info: what the model is, its stored metrics, and whether any test individual was in its training data
- cluster_consistency: per haploblock, whether train and test individuals carry the same clusters, with the same frequencies, compared against random splits of the same size
- gwas_enrichment: which haploblocks contain more GWAS hits than expected, plus an overall test

New kinds of checks go in a Python file loaded with `--plugins` my_checks.py; the script's header shows a short example. Results are written to report.md, summary.json and one CSV per task in `<graph-dir>/validation_<model>/`.

#### Getting GWAS data

gwas_enrichment reads any text file with chromosome, position and p-value columns, and recognises the GIANT and GWAS Catalog column names automatically. The positions must use the same genome build as the haploblock regions.

GIANT height (Yengo et al. 2022). Download a summary statistics file, e.g. Yengo.2022.height.GWAS.ALL (all ancestries) or .EUR, from the GIANT consortium data files page. The columns are SNPID RSID CHR POS EFFECT_ALLELE OTHER_ALLELE EFFECT_ALLELE_FREQ BETA SE P N, with positions in hg19/GRCh37. The file can be used directly if the haploblocks are hg19. For GRCh38, lift the positions over first, e.g. with CrossMap and the UCSC hg19ToHg38.over.chain.gz chain file:

```bash
zcat Yengo.2022.height.GWAS.ALL.gz | awk 'NR>1 {OFS="\t"; print "chr"$3, $4-1, $4, $2, $10}' > height_hg19.bed
CrossMap bed hg19ToHg38.over.chain.gz height_hg19.bed height_hg38.bed
(printf "CHR\tPOS\tRSID\tP\n"; awk 'BEGIN{OFS="\t"} {print $1, $3, $4, $5}' height_hg38.bed) > height_hg38.tsv
```

GWAS Catalog (any trait). Download all associations from the GWAS Catalog downloads page, e.g. gwas-catalog-associations_ontology-annotated-full.zip. It contains only reported hits, in GRCh38 coordinates (columns CHR_ID, CHR_POS, SNPS, P-VALUE), so the per-region test then uses region length as the background. Keep one trait:

```python
import pandas as pd
cat = pd.read_csv("gwas-catalog-associations_ontology-annotated-full.zip", sep="\t", dtype=str)
height = cat[cat["MAPPED_TRAIT"].str.contains("body height", case=False, na=False)]
height[["CHR_ID", "CHR_POS", "SNPS", "P-VALUE"]].to_csv("gwas_catalog_height.tsv", sep="\t", index=False)
```

Full summary statistics (like GIANT) give the stronger per-region test, because each haploblock is compared against the number of SNPs tested in it.