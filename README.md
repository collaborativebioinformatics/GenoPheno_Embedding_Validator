# Haplograph → Evo 2

**Which way of turning genotype–phenotype data into vectors works best for downstream models, and when?**

This repository benchmarks embedding strategies for haploblock-based genotype–phenotype data. The same
haploblock data are encoded several ways (flat binary strings, graph embeddings from a haplograph, and later
metagraph embeddings) and fed into two kinds of downstream models: **XGBoost** and the **Evo 2 (7B)** DNA
foundation model. We compare them first on germline data (1000 Genomes), then on paired **tumor vs. adjacent
normal (TAN)** data, and then with **proteomics** added to the genomic layer.

## Background

This work builds on the hash-based genotype–phenotype mapping framework from the
["Beyond Beacons" BioHackrXiv preprint](https://osf.io/preprints/biohackrxiv/xhkc3_v1). That framework defines
genomic hashes from haplotype-resolved variant sets within recombination-defined haploblocks. It is implemented
in the [Haploblock_Clusters_ElixirBH25 pipeline](https://github.com/collaborativebioinformatics/Haploblock_Clusters_ElixirBH25).
See also the live project site at [haploblocks.org](https://haploblocks.org).

What has been done so far:

- Established and optimized the haploblock method for discretizing genomic data.
- Processed the 1000 Genomes data set with the haploblock method (paper submitted).
- Generated binary strings from haploblock clusters and tested linear models, XGBoost and deep learning on them.
  Results were promising but not conclusive.
- Established and optimized the **haplograph** method, a graph of co-occurring haploblock clusters
  (manuscript in preparation).

## Questions we are testing

1. When is a simple clustering method (linclust) enough, and when is a graph needed?
2. Which representation of haploblocks works best, and in which setting?
   - binary strings of haploblocks fed directly into a DNA foundation model;
   - XGBoost to find coherent groups of haploblocks;
   - a haplograph embedded with node2vec;
   - a haplograph embedded with a GNN (PyTorch Geometric) to isolate subnetworks and build contextualized vectors;
   - XGBoost on haplograph-derived features.
3. Do the answers hold in both **germline disease** and **cancer**?
4. When is **early fusion** or **late fusion** of proteomics with genomic and phenotype data more effective?

## Projects

| # | Project | Goal | Status |
|---|---|---|---|
| 1 | XGBoost embeddings | Compare graph, linear and (stretch) metagraph embeddings as XGBoost inputs | In progress (steps 1–3 below) |
| 2 | Evo 2 7B | Run the same graph vs. string comparison through Evo 2 7B | In progress (step 4 below) |
| 3 | Tumor / adjacent normal (TAN) | Build metagraph haplographs (MHGs) for paired tumor and normal tissue | In progress |
| 5 | Proteomics integration | Test early vs. late fusion of proteomics with haploblocks, via ProGenome, into Evo 2 | Data prep done ([`proteomics_prep/`](proteomics_prep/)) |

### 1. XGBoost embeddings

Compares how useful different embedding strategies are for training and tuning XGBoost:

- **Graph embeddings**: node embeddings from genotype–phenotype graphs (haplographs) made with PyTorch Geometric
  (GNN) or node2vec, and possibly combined with other data types.
- **Linear embeddings**: baseline string- or sequence-based representations of the same haploblocks, for
  comparison with the graph-based approaches.
- **Metagraph embeddings (stretch goal)**: embeddings of graphs that combine different graph types (for example
  genomes and literature). These need specialized software, because disparate graphs tend to split apart rather
  than integrate cleanly when simply merged.

### 2. Evo 2 7B

Runs the same graph vs. string comparison through the Evo 2 7B foundation model, alongside XGBoost. Node
embeddings replace Evo 2's nucleotide token embeddings, so each individual is presented to the model as the
ordered list of haploblock clusters they carry (see step 4 under Usage).

### 3. Tumor vs. adjacent normal (TAN)

Germline data such as 1000 Genomes give one genome per person. In cancer, each patient has two related genomes:
the tumor and the adjacent normal tissue taken at the same surgery. The paired design lets each patient act as
their own control. Germline background is the same in both samples, so differences between them at the
haploblock level point to somatic events, such as allelic imbalance, copy-number change or loss of one haplotype
in a block.

Goals:

- **Build metagraph haplographs (MHGs) for TAN data.** Patients, tumor and normal samples, haploblocks and
  clusters, and the tumor-minus-normal differences become nodes and edges of one graph. We reuse the existing
  "nucleate" code to build the MHGs.
- **Build the haplograph for the whole genome** rather than chromosome by chromosome, so that events spanning
  chromosomes and arms are captured in one graph.
- **Embed the MHGs with the Project 1 methods** (node2vec, GNN, XGBoost, linear baselines) and compare them with
  the same downstream models.
- **Ask the small-cohort question honestly.** Paired TAN cohorts are small, which makes training a GNN from scratch
  hard. We test whether embeddings carried in from larger graphs or foundation models, and graph structure itself,
  help in this setting.
- **Benchmark against known answers.** Synthetic paired cohorts with planted signals (known blocks and known
  marker proteins) let us measure whether each method recovers what was planted before we apply it to real data.

Real patient data are never stored in this repository.

### 5. Proteomics integration

Tests whether using [ProGenome](https://github.com/collaborativebioinformatics/ProGenome) to load proteomic data
onto haploblocks, and from there into Evo 2, is more effective than modeling proteomics on its own. ProGenome
labels each haploblock with the proteins encoded in it, so a protein signal can be traced back to the region of
the genome it comes from.

The main comparison is **early vs. late fusion**:

- **Early fusion**: proteins become nodes in the graph, attached to the haploblock that holds their gene. The graph
  is embedded once, with genomic and protein information together.
- **Late fusion**: the protein data (tumor minus normal per protein and patient) are embedded separately, and the
  protein and genomic embeddings are combined at the model level.

We compare both with genomics only and proteomics only, on the synthetic paired cohorts (with planted marker
proteins) and then on real paired tumor/normal proteomics.

[`proteomics_prep/`](proteomics_prep/) holds the R scripts that prepare paired tumor/normal proteomics for these
tests. They place each protein in the haploblock that holds its gene's transcription start site, compute tumor
minus normal per protein and patient, and write node and edge CSVs (patients, blocks, proteins, protein-to-block
edges) for early fusion, plus a protein × patient matrix for late fusion. Each block also carries its haplograph id
(`chr21_14215892-14284114`), so proteins can be joined to the cluster nodes from `1_build_graph.py` by prefix.
See [`proteomics_prep/README.md`](proteomics_prep/README.md).

## Proof of concept: 1000 Genomes haplograph → Evo 2

As a proof of concept, the scripts below build a haploblock co-occurrence graph from the
[1000 Genomes haplograph](https://data.haploblocks.org/haplograph/1000G/), turn its nodes into embeddings
(node2vec, GNN or XGBoost), and fine-tune Evo 2 (7B) on those embeddings.

## Dependencies

- Python 3.11 or 3.12, Linux, NVIDIA GPU with CUDA 12.1+ (80 GB recommended for step 4)
- numpy, pandas, scipy (all steps)
- torch (node2vec and GNN embeddings, Evo 2)
- xgboost (XGBoost embeddings)
- evo2 + flash-attn (step 4)
- R 4.3+ with tidyverse, optparse and cowplot (`proteomics_prep/` only)

## Install

```bash
pip install numpy pandas scipy torch xgboost
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
```

## Usage

Run the scripts in order. Step 1 writes the graph to `haplograph/genome/`; steps 2–4 read and write there (change
with `--graph-dir`).

```bash
python 1_build_graph.py                            # whole genome (or --chroms chr21,chr22)
python 2_ancestry_labels.py                        # needed for xgboost or --objective ancestry
python 3_node_embeddings.py --method node2vec      # or gnn / xgboost
python 4_finetune_evo2.py   --method node2vec      # --objective next (default) or ancestry
```

Add `--chroms chr21,chr22` to any step for a quick test run, and `--help` to see all options.

## License

MIT (see [`LICENSE`](LICENSE)).
