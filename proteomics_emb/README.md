# proteomics_emb: Generate embeddings of the proteomics data with ProLM for latefusion

We use the code from [ProLM](https://github.com/Qiu-Shizheng/ProLM).

The file "finetune.py" contains the following changes as compared with the main branch of ProLM:

- saving the embeddings obtained from the backbone model,
- saving the backbone model itself.

So far this has been tested with the example as provided in the documentation of ProLM.

Next we will test that this works with the data from our project, first of all "synthtan_prolm".

Note: running the fine tuning requires the ProLM package to be installed in the python environment, so we may need to fork that package with the changes proposed here.


## Usage

Needs Python 3 with `torch` and `pandas` .

```bash
python -m prolm.finetune \
  --expression-csv path/to/expression.csv \
  --labels-csv path/to/labels.csv \
  --protein-embeddings path/to/protein_embeddings.npz \
  --output-dir path/to/output/ \
  --epochs 2 \
  --batch-size 4 \
  --hidden-size 32 \
  --num-layers 1 \
  --chunk-size 16
```

## Output

Note: the path and file names below have not been generated yet!

| Folder | Contents |
|---|---|
| `synthtan_prolm_outputs/` | `prolm_backbone_finetuned.pt`, `prolm_embeddings.pt` |


