# Haplograph → Evo 2

Builds a genome-wide (or per-chromosome-selection) haploblock co-occurrence graph from the [1000 Genomes haplograph](https://data.haploblocks.org/haplograph/1000G/), turns its nodes into embeddings with node2vec, a GNN or XGBoost, fine-tunes Evo 2 (7B) on those embeddings using a training half of the individuals, and validates the fine-tuned model on the held-out half.

For a full reference (what the haplograph is, every script's inputs, outputs and main options, how to extend the pipeline, and its current limitations), see [DOCUMENTATION.md](DOCUMENTATION.md).

## Dependencies

- Python 3.11 or 3.12, Linux, NVIDIA GPU with CUDA 12.1+ (80 GB recommended for step 4)
- numpy, pandas, scipy (all steps)
- torch (node2vec and GNN embeddings, Evo 2, reading models in step 5)
- xgboost (XGBoost embeddings)
- evo2 + flash-attn (step 4)

## Install

```bash
pip install numpy pandas scipy torch xgboost
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
```

## Usage

The pipeline uses the [BH_26 train/test split](https://data.haploblocks.org/BH_26/splits/). `half_A.txt` (1,252 individuals) is the train set, used by steps 1–4. `half_B.txt` (1,251 individuals) is the test set, which only step 5 uses. Both are plain text, one 1000 Genomes sample ID per line.

Pass the train half once, to step 1. It builds the graph from those individuals only and writes `train_individuals.txt` into the graph folder, and steps 2–4a then use it automatically:

```bash
SPLITS=https://data.haploblocks.org/BH_26/splits
G=haplograph/genome_train_half_A

python 1_build_graph.py --train-individuals $SPLITS/half_A.txt      # whole genome (or --chroms chr21,chr22)
python 2_ancestry_labels.py --graph-dir $G                           # needed for xgboost or --objective ancestry
python 3_node_embeddings.py --method node2vec --graph-dir $G         # or gnn / xgboost
python 4a_finetune_evo2_embeddings.py --method node2vec --graph-dir $G
python 5_validate_evo2.py --model $G/evo2_next_node2vec.pt --graph-dir $G --tasks validation_tasks.txt
```

What each step takes from the train half:
- **Step 1:** the edge weights, cluster support and lift come from training individuals only. The carrier matrix still holds everyone, because step 5 needs the test individuals' clusters.
- **Step 2:** node ancestry labels.
- **Step 3:** the carrier features for GNN and XGBoost.
- **Step 4a:** trains and validates within the train half only. The checkpoint records which individuals it trained on.

Each step checks that it uses the same training set as the step before and stops if not. Every script also accepts `--train-individuals` to override the graph's file. Leaving it out of step 1 builds the graph from all individuals, as before.

If the graph names individuals differently from the sample IDs (e.g. `IND1, IND2…`), add `--sample-map map.csv` (two columns: graph name, sample ID) to the step that reads the split file.

For a quick test, add `--chroms chr21,chr22` to step 1 and use `G=haplograph/chr21_chr22_train_half_A`. Each script's header describes its input and output file formats, and `--help` lists all options.

### Optional, experimental: step 4a with NVIDIA BioNeMo

`4a_finetune_evo2_embeddings_bionemo.py` does the same fine-tuning as step 4a, but loads and runs Evo 2 through [NVIDIA BioNeMo](https://docs.nvidia.com/bionemo-framework/latest/models/evo2/) (NeMo 2 / Megatron Core) instead of the `evo2` package. It uses the training set the same way. It has not been tested on a GPU yet and may need adjusting for your BioNeMo version (the affected parts are marked `VERSION-SENSITIVE` in the code).

It runs inside the BioNeMo Framework container (`nvcr.io/nvidia/clara/bionemo-framework`) and needs an Evo 2 checkpoint in NeMo2 format:

```bash
evo2_convert_to_nemo2 --model-path hf://arcinstitute/savanna_evo2_7b_base --model-size 7b --output-dir nemo2_evo2_7b_8k
python 4a_finetune_evo2_embeddings_bionemo.py --method gnn --graph-dir $G --ckpt-dir nemo2_evo2_7b_8k
```

### Hash-phenotype fine-tuning

To fine-tune on hash-phenotype associations instead, use step 4b on its own (no steps 1–3 needed). Its input is a TSV with two columns, a genomic hash (0/1 string) and a phenotype (e.g. height). Use a TSV built from the train half only, so the test half stays unseen.

```bash
python 4b_finetune_evo2_hash.py --hash-pheno hash_height.tsv   # numeric -> regression, labels -> classification
```

## Step 5: validation

`5_validate_evo2.py` checks one fine-tuned model (any `.pt` from step 4) against the test set. By default it reads `half_A.txt` (train) and `half_B.txt` (test) from the BH_26 split, downloading them automatically. Or download them once and pass the folder:

```bash
wget -r -np -nH --cut-dirs=2 -R "index.html*" https://data.haploblocks.org/BH_26/splits/
python 5_validate_evo2.py --model $G/evo2_next_gnn.pt --graph-dir $G --splits splits/ --gwas height_hg38.tsv
```

It also checks for leakage at two levels:
- It warns if any test individual is in the graph's training set (step 1).
- It warns if any test individual is in the model's training set (step 4a).

Graph individuals in neither half are left out and counted in the report. Other split files can be given with `--train-individuals` / `--test-individuals`, and `--splits none` uses the model's own held-out individuals instead.

The checks to run are listed in a tasks file (`validation_tasks.txt` is an example), one per line with `key=value` parameters. Add a line to run a check again with other settings. `--list-tasks` shows every task and its parameters.

- `model_info`: what the model is, its stored metrics, and whether any test individual was in its training data
- `cluster_consistency`: per haploblock, whether train and test individuals carry the same clusters, with the same frequencies, compared against random splits of the same size
- `gwas_enrichment`: which haploblocks contain more GWAS hits than expected, plus an overall test

New kinds of checks go in a Python file loaded with `--plugins my_checks.py`; the script's header shows a short example. Results are written to `report.md`, `summary.json` and one CSV per task in `<graph-dir>/validation_<model>/`.

### Getting GWAS data

`gwas_enrichment` reads any text file with chromosome, position and p-value columns, and recognises the GIANT and GWAS Catalog column names automatically. **The positions must use the same genome build as the haploblock regions.** The haplograph README doesn't state its build, so check it before running.

**GIANT height (Yengo et al. 2022).** Download a summary statistics file, e.g. `Yengo.2022.height.GWAS.ALL` (all ancestries) or `.EUR`, from the [GIANT consortium data files page](https://giant-consortium.web.broadinstitute.org/GIANT_consortium_data_files). The columns are `SNPID RSID CHR POS EFFECT_ALLELE OTHER_ALLELE EFFECT_ALLELE_FREQ BETA SE P N`, with positions in **hg19/GRCh37**. The file can be used directly if the haploblocks are hg19. For GRCh38, lift the positions over first, e.g. with [CrossMap](https://crossmap.sourceforge.net/) and the UCSC `hg19ToHg38.over.chain.gz` chain file:

```bash
zcat Yengo.2022.height.GWAS.ALL.gz | awk 'NR>1 {OFS="\t"; print "chr"$3, $4-1, $4, $2, $10}' > height_hg19.bed
CrossMap bed hg19ToHg38.over.chain.gz height_hg19.bed height_hg38.bed
(printf "CHR\tPOS\tRSID\tP\n"; awk 'BEGIN{OFS="\t"} {print $1, $3, $4, $5}' height_hg38.bed) > height_hg38.tsv
```

**GWAS Catalog (any trait).** Download all associations from the [GWAS Catalog downloads page](https://www.ebi.ac.uk/gwas/docs/file-downloads), e.g. `gwas-catalog-associations_ontology-annotated-full.zip`. It contains only reported hits, in GRCh38 coordinates (columns `CHR_ID`, `CHR_POS`, `SNPS`, `P-VALUE`), so the per-region test then uses region length as the background. Keep one trait:

```python
import pandas as pd
cat = pd.read_csv("gwas-catalog-associations_ontology-annotated-full.zip", sep="\t", dtype=str)
height = cat[cat["MAPPED_TRAIT"].str.contains("body height", case=False, na=False)]
height[["CHR_ID", "CHR_POS", "SNPS", "P-VALUE"]].to_csv("gwas_catalog_height.tsv", sep="\t", index=False)
```

Full summary statistics (like GIANT) give the stronger per-region test, because each haploblock is compared against the number of SNPs tested in it.