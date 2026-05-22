# DWDMixer

## Datasets

You can download all the datasets from [TimeMixer](https://drive.usercontent.google.com/download?id=1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP&export=download&authuser=0).
All the datasets are well pre-processed and can be used easily.

## How To Use
```
conda create -n DWDMixer python=3.8    
```
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia 
```
```
pip install -r requirements.txt  
```
## Train DWDMixer
We provide the experiment scripts of all benchmarks under the folder ./scripts. You can reproduce the experiment results by:
```
bash ./scripts/long_term_forecast/ETT_script/DWDMixer_ETTh1.sh
bash ./scripts/long_term_forecast/ECL_script/DWDMixer.sh
bash ./scripts/long_term_forecast/Traffic_script/DWDMixer.sh
bash ./scripts/long_term_forecast/Solar_script/DWDMixer.sh
bash ./scripts/long_term_forecast/Weather_script/DWDMixer.sh
bash ./scripts/short_term_forecast/PEMS/DWDMixer.sh
```

## Acknowledgement
We appreciate the following GitHub repos a lot for their valuable code and efforts.
[TimeMixer](https://github.com/kwuking/TimeMixer)

