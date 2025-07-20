# DynamicCLS for ECG Diagnosis

This is a lightweight and efficient transformer model that has dynamic receptive fields throughout layers, especially suitable for multi-channel time-sequential data classification like ECG signals.  
This paper was accepted by MIDL 2025 as a short paper. Feel free to check more details in our 📄 Paper

📌 Check out our paper:  
**[Dynamic Scale for Transformer](https://openreview.net/pdf?id=vWkjFvYUws)**  

---
Our model is built upon the champion **['prna'](https://www.cinc.org/archives/2020/pdf/CinC2020-107.pdf)** of the PhysioNet2020 Challenge, a plain transformer architecture. Thanks to their great pioneering work.

To reproduce our results on the PhysioNet2020 Challenge data, you need to download their data from their **[website](https://moody-challenge.physionet.org/2020/)** , **OR** follow the instructions below. 

This model is an updated version of a Three-stage transformer in another paper by us. You can also refer to our repository **[3stageFormer](https://github.com/xiaoyatang/3stageFormer.git)**.

Built upon the Three-stage transformer, we implement a dynamic receptive field in 'model.py'. Other settings remain the same. 


### 📁 `feats` Directory Setup. Step 1: Download Required Files(this step is to prepare necessary files used by 'prna')

Download the required folders from the shared Google Drive link:

👉 [Download feats folders from Google Drive](https://drive.google.com/drive/folders/1XWfkR159jWJCcC6jJ9DQECq4XV-of8JG?usp=sharing)

Simply place the downloaded folders inside a local directory named `feats`.

---

### 📂 Step 2: Directory Structure

Your `feats/` directory should look like this:
```text
feats/
├── CPSC-Extra/
├── Georgia/
├── PTB/
├── PTB-XL/
├── StPetersburg/
├── pyeeg/
├── utils/
├── __init__.py
├── dataset.py
├── feature_map.py
├── get_feats.py
└── signal_process.py
```
### 📂 Step 3: Data Downloading
Download the data from the shared Google Drive link:

👉 [Download data folders](https://drive.google.com/file/d/1X9ORozkSYE0NGX8GdyKgzVENX5NOsu7e/view?usp=sharing)

Make an empty folder 'data', put it under the main directory. Unzip the downloaded file and place it under 'data'.

If you are using your own dataset, you can skip this step and specify the path to your data. 

## 🛠 Step 4: Environment Setup

To run our model, please create the conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml -n your_env_name
conda activate your_env_name
```
## 🛠 Step 5:  Running
```bash
python train_model.py ./data/classification-of-12-lead-ecgs-the-physionetcomputing-in-cardiology-challenge-2020-1.0.2/training path_to_your_output
```
---
## 📚 Citation

If you find this repository useful, please consider giving a star ⭐ and citing our work 🩺:

```bibtex
@inproceedings{tang2025dynamic,
  title={Dynamic Scale for Transformer},
  author={Tang, Xiaoya and Li, Xiwen and Tasdizen, Tolga},
  booktitle={Medical Imaging with Deep Learning-Short Papers},
  year={2025}
}
