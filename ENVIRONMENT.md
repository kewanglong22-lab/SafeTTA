# Environment

The public SafeTTA release retains the **authoritative frozen pip snapshot**
used by the validated author environment and derives the root environment lock
directly from that snapshot.

## Recommended recreation

```bash
conda env create -f environment.yml
conda activate safetta
```

`environment.yml` contains **100 pinned pip requirements** from
`environment/environment_freeze.txt`.

Selected validated packages:

- Python: 3.10.20
- torch==2.5.1+cu121
- torchvision==0.20.1+cu121
- numpy==1.26.4
- pandas==2.3.3
- scipy==1.15.3
- scikit-learn==1.7.2
- transformers==5.15.0
- joblib==1.5.3

The repository also contains:

- `environment/environment_freeze.txt` — authoritative frozen pip snapshot.
- `environment/environment_summary.json` — compact machine-readable environment metadata.
- `requirements-lock.txt` — root copy of the frozen pip requirements.
- `environment_minimal.yml` — optional lightweight replay convenience environment.

`environment_minimal.yml` is **not** the authoritative environment provenance;
use the full `environment.yml` when reproducing the validated dependency set.

If the CUDA 12.1 PyTorch wheels are unsuitable for a target system, installing
a platform-appropriate PyTorch build is a platform adaptation rather than an
exact recreation of the validated author environment.

The public reproducibility claim concerns the 44 paper-level numeric anchors.
Fresh wall-clock runtime remains hardware dependent.
