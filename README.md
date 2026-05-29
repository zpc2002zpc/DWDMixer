# 🌊 DWDMixer

> **DWDMixer: Detail-Aware Wavelet Decomposition and Mixing for Time Series Forecasting in IoT Systems**

## 📖 Citation

If you find this work helpful for your project, please consider citing the following paper 😊

```bibtex
@article{zhang2026dwdmixer,
  title={DWDMixer: Detail-Aware Wavelet Decomposition and Mixing for Time Series Forecasting in IoT Systems},
  author={Zhang, Pengcheng and Huang, Wei and Wang, Jie and Peng, Ran and Zhai, Qiang and Ouyang, Xiaocao},
  journal={IEEE Internet of Things Journal},
  year={2026},
  publisher={IEEE}
}
```

⭐ If you find this repository useful, please consider giving it a star!

## 📂 Datasets

You can download all the datasets from [TimeMixer](https://drive.usercontent.google.com/download?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download&authuser=0).

📌 All datasets are well pre-processed and can be used directly.

## ⚙️ How To Use

### 1️⃣ Create Environment

```bash
conda create -n DWDMixer python=3.8
```

### 2️⃣ Install PyTorch

```bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Train DWDMixer

We provide experiment scripts for all benchmarks under the `./scripts` folder.

You can reproduce the experimental results by running:

```bash
bash ./scripts/long_term_forecast/ETT_script/DWDMixer_ETTh1.sh
bash ./scripts/long_term_forecast/ECL_script/DWDMixer.sh
bash ./scripts/long_term_forecast/Traffic_script/DWDMixer.sh
bash ./scripts/long_term_forecast/Solar_script/DWDMixer.sh
bash ./scripts/long_term_forecast/Weather_script/DWDMixer.sh
bash ./scripts/short_term_forecast/PEMS/DWDMixer.sh
```

## 🙏 Acknowledgement

We sincerely appreciate the following GitHub repository for its valuable codebase and efforts:

🔗 [TimeMixer](https://github.com/kwuking/TimeMixer)
