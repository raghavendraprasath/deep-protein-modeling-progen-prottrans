# Deep Learning for Protein Sequence Modeling using ProGen and ProtTrans

This project replicates the ProGen and ProtTrans research papers, applying transformer-based language models to protein sequence modeling. It includes conditional sequence generation, embedding visualization, downstream task evaluation, and empirical analysis using simulated datasets.

---

## 📚 Project Overview

This project was developed as part of the course **Data Science Engineering Methods and Tools (INFO 6105)** at **Northeastern University** under the guidance of **Professor Akash Murthy**.

The objective was to replicate two foundational papers:

- [ProGen: Language Modeling for Protein Generation (Madani et al.)](https://arxiv.org/abs/2004.03497)  
  [Official GitHub](https://github.com/salesforce/progen)

- [ProtTrans: Cracking the Language of Life's Code with Self-Supervised Deep Learning (Elnaggar et al.)](https://arxiv.org/abs/2007.06225)  
  [Official GitHub](https://github.com/agemagician/ProtTrans)

---

## 🎥 Project Video Presentation

[![Watch the Video](https://img.youtube.com/vi/q7PJ2QnCUIA/0.jpg)](https://www.youtube.com/watch?v=q7PJ2QnCUIA)

> Click above to watch the presentation summarizing our work.

---

## 🛠️ Implementation Details

- Language: **Python** (Google Colab GPU used for experiments)
- Libraries: TensorFlow, PyTorch, Hugging Face Transformers, Matplotlib, Seaborn, Scikit-learn
- Visualization & Analysis: **R** (ggplot2, Rtsne, umap, pheatmap)
- Embedding Extraction, Sequence Generation, Clustering, and Evaluation Tasks

---

## 📂 Repository Structure

```
deep-protein-modeling-progen-prottrans/
├── Base Research Papers/
│   ├── progen_paper.pdf
│   └── prottrans_paper.pdf
│
├── Plots/
│   ├── Python Notebook Plots/
│   │   ├── ProGen/
│   │   │   ├── progen_secondary_structure_accuracy_conditioning_tags.png
│   │   │   ├── progen_conformational_energies_mutations.png
│   │   │   ├── progen_generated_sequence_loglikelihood.png
│   │   ├── ProtTrans/
│   │       ├── prottrans_confusion_matrix_localization.png
│   │       ├── prottrans_cosine_similarity_embeddings.png
│   │       ├── prottrans_pca_projection_embeddings.png
│   │       ├── prottrans_umap_projection_embeddings.png
│
│   ├── R Plots/
│   │   ├── ProGen/
│   │   │   ├── Rplotprogen_sequence_clustering.png
│   │   │   ├── progen_log_likelihood_scores.png
│   │   │   ├── progen_secondary_structure_accuracy.png
│   │   │   ├── progen_sequence_similarity_histogram.png
│   │   ├── ProtTrans/
│   │       ├── prottrans_confusion_matrix.png
│   │       ├── prottrans_cosine_similarity_heatmap.png
│   │       ├── prottrans_tsne_embeddings.png
│   │       ├── prottrans_umap_embeddings.png
│
├── Final Project Presentation.pptx
├── Final Project Report.pdf
├── GitHub Repo Link.txt
├── ProGen Implementation.ipynb
├── ProtTrans Implementation.ipynb
├── README.md
├── YouTube Video Link.txt
```

## 📈 Key Plots and Visualizations

### ProGen Model (Python Notebook Plots)

#### Secondary-Structure Accuracy vs Conditioning Tags
![Secondary Structure Accuracy](Plots/Python%20Notebook%20Plots/ProGen/progen_secondary_structure_accuracy_conditioning_tags.png)
> Accuracy improves with more conditioning tags used in ProGen sequence generation.

#### Conformational Energies: ProGen vs Mutation Baselines
![Conformational Energies](Plots/Python%20Notebook%20Plots/ProGen/progen_conformational_energies_mutations.png)
> Boxplot comparing conformational energy deviations across different mutation baselines.

#### Log-Likelihood Scores of Generated Sequences
![Log-Likelihood Scores](Plots/Python%20Notebook%20Plots/ProGen/progen_generated_sequence_loglikelihood.png)
> Log-likelihood distribution across generated sequences showing plausibility levels.

---

### ProtTrans Model (Python Notebook Plots)

#### Confusion Matrix: Protein Localization Prediction
![Confusion Matrix Localization](Plots/Python%20Notebook%20Plots/ProtTrans/prottrans_confusion_matrix_localization.png)
> Confusion matrix showing classification performance based on embeddings.

#### Cosine Similarity Heatmap of Embeddings
![Cosine Similarity Heatmap](Plots/Python%20Notebook%20Plots/ProtTrans/prottrans_cosine_similarity_embeddings.png)
> Cosine similarity visualization between protein embeddings extracted using ProtTrans.

#### PCA Projection of Protein Embeddings
![PCA Projection](Plots/Python%20Notebook%20Plots/ProtTrans/prottrans_pca_projection_embeddings.png)
> PCA projection showing separation of protein classes based on embeddings.

#### UMAP Projection of Protein Embeddings
![UMAP Projection](Plots/Python%20Notebook%20Plots/ProtTrans/prottrans_umap_projection_embeddings.png)
> UMAP projection illustrating tighter neighborhood structures among protein classes.

---

## ⚙️ How to Reproduce

1. Clone this repository.
2. Open the `.ipynb` notebooks in Google Colab.
3. Ensure GPU runtime is enabled (Runtime > Change runtime type > GPU).
4. Run all cells to reproduce sequence generation, embedding extraction, clustering, and evaluation.
5. Refer to the `/Plots/` folder for all visualizations.

---

## ✍️ Team Members

- Raghavendra Prasath Sridhar
- Janhavi Vijay Patil
- Gunashree Rajakumar

---

## 📜 Acknowledgements

Special thanks to **Professor Akash Murthy** and **TA Kiran Sathya Sunkoji Rao** for their valuable guidance and feedback throughout the course.

---
