<p align="center">
  <h1 align="center">GSplatLoc: Ultra-Precise Camera Localization via 3D Gaussian Splatting</h1>
  <p align="center">
    <a href="https://notes.atticux.me/"><strong>Atticus Zeller</strong></a>
  </p>
</p>

<p align="center">
  <a href="">
    <img src="./docs/flowchat.png" width="100%">
  </a>
</p>

## ⚙️ Setting Things Up

> CUDA 12.1 [download](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local)

Install [UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then setup environment from the `pyproject.toml` :

```bash
uv sync --all-extras --all-groups --dev
source .venv/bin/activate
```

## 🔨 Running GsplatLoc

Downloading the Data:

```bash
bash scripts/download_datasets.sh
```

Initialize the Weights & Biases logging:
1. use wandb login with api key https://wandb.ai/authorize, then `wandb login --relogin`
2. update `.env` with your wandb content.

Reproducing Results:

```bash
export MPLBACKEND=Agg
bash scripts/run_eval.sh
```

## 📌 Citation

If you find our paper and code useful, please cite us:

```bib
@misc{zeller2024gsplatlocultraprecisecameralocalization,
      title={GSplatLoc: Ultra-Precise Camera Localization via 3D Gaussian Splatting},
      author={Atticus J. Zeller},
      year={2024},
      eprint={2412.20056},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.20056},
}
```
