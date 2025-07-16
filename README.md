# DynamicCLS for ECG Diagnosis

This is a lightweight and efficient transformer model that has dynamic receptive fields throughout layers, especially suitable for multi-channel time-sequential data classification like ECG signals.  
This paper was accepted by MIDL 2025 as a short paper. Feel free to check more details in our 📄 Paper

📌 Check out our paper:  
**[Dynamic Scale for Transformer](https://openreview.net/pdf?id=vWkjFvYUws)**  

---
## 🛠 Environment Setup

To run our model, please create the conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate prna-oldlib

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
