# proteomics_emb: Generate embeddings of the proteomics data with ProLM for latefusion

We use the code from [ProLM](https://github.com/Qiu-Shizheng/ProLM).

The file "finetune.py" contains the following changes as compared with the main branch of ProLM:

- saving the backbone model itself (`backbone.pt`, at the best epoch);
- saving one embedding per patient (`patient_embeddings.csv`: `eid`, `e0` ... `e<hidden_size - 1>`), computed once
  after training from the selected model, in evaluation mode (no dropout, no gradients), in batches and on the
  model's device;
- when no `--pretrained-checkpoint` is given, all parameters are trained. ProLM's `build_model` always calls
  `freeze_backbone`, which otherwise leaves the encoder, the feature and expression projections and the [CLS]
  token at their random initial values (only the classifier head would train).

**Shape of the embeddings.** The backbone splits the proteins into chunks of `--chunk-size`, runs each chunk with
its own [CLS] token and averages the chunk [CLS] vectors. The result is one vector per patient,
`(n_patients, hidden_size)`, which is what late fusion needs. Per-protein vectors are not returned by ProLM; getting
them would need a change to `model.py`.

**Still to handle for a fair fusion comparison.** `finetune.py` makes its own train/test split (`--test-size`) and
picks the best epoch on that test split. The patient embeddings of the training patients are therefore fitted to
their labels. For late fusion, train ProLM inside each fold of the downstream evaluation and take the embeddings
from that fold's model.

So far this has been tested with the example as provided in the documentation of ProLM.

Next we will test that this works with the data from our project, first of all "synthtan_prolm".

Note: `finetune.py` uses relative imports (`from .data import ...`), so it runs only from inside the ProLM package:
install a fork of ProLM in editable mode (`pip install -e .`) with this file as `src/prolm/finetune.py`, then run
`python -m prolm.finetune`.


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
  --chunk-size 512
```

## Output

| Folder | Contents |
|---|---|
| `--output-dir` | `best_classifier.pt`, `backbone.pt`, `patient_embeddings.csv`, `metrics.json`, `protein_order.txt` |


