
# DFH-ViT: Deep Fossil Hierarchical Vision Transformer

**A Graduate Course Project for "Intelligent Software Engineering" at UCAS**

This project implements a **Deep Fossil Hierarchical Vision Transformer (DFH-ViT)** for classifying microscopic fossil foraminifera. It was developed as part of the **Intelligent Software Engineering** graduate course at the University of Chinese Academy of Sciences (UCAS), taught by **Professor Tiejian Luo**.

## 📋 Project Overview & Context

The "Intelligent Software Engineering" course at UCAS focuses on new challenges in software engineering and the development of new technologies to address them, including software design, architecture, and quality[citation:6]. This project directly applies course principles to design an intelligent system for a scientific domain problem.

The **DFH-ViT** model combines hierarchical classification, synthetic data generation, and active learning to address key challenges in paleontological research: **limited labeled data**, **hierarchical taxonomy**, and the need for **efficient expert annotation**.

## 👨‍🏫 Instructor Information
*   **Course Instructor**: Professor **Tiejian Luo**
*   **Affiliation**: School of Information Science and Engineering, University of Chinese Academy of Sciences (UCAS).
*   **Email**: `tjluo@gucas.ac.cn`
    
## 🧠 Technical Foundation: Vision Transformer (ViT)

This project is built upon the **Vision Transformer (ViT)** architecture, a groundbreaking approach that adapts the Transformer model—originally successful in Natural Language Processing (NLP)—for computer vision tasks[citation:2][citation:8].

**Key ViT Concepts Used in This Project:**

1.  **Image Patching**: An input image is split into fixed-size patches (e.g., 16x16 pixels), which are treated as a sequence of "visual tokens," similar to words in a sentence[citation:8].
2.  **Patch Embedding & Positional Encoding**: Each patch is flattened and projected into a vector (embedding). Learnable positional encodings are added to retain spatial information[citation:8].
3.  **Transformer Encoder**: The sequence of patch embeddings is processed by a standard Transformer encoder stack, which uses **Multi-Head Self-Attention** to model relationships between all patches globally[citation:8].
4.  **Classification Head**: A special `[CLS]` token gathers global information, and its final state is fed into a Multi-Layer Perceptron (MLP) for classification[citation:8].

**Why ViT?**
Compared to traditional Convolutional Neural Networks (CNNs), ViTs excel at capturing **long-range dependencies** across an entire image due to their global attention mechanism[citation:8]. While often data-hungry, they are highly flexible and scalable, making them suitable for complex tasks like hierarchical fossil classification when combined with strategies like synthetic data and transfer learning[citation:8].

## 🚀 Key Features of DFH-ViT

1.  **Hierarchical Classification**: Two-level taxonomy (coarse morphological groups → fine species) using a custom hierarchical head architecture.
2.  **Synthetic Data Generation**: A `SyntheticFossilFractalDataset` class generates fractal-based simulations of foraminifera (spumellarian/nassellarian morphologies) for model pretraining, addressing data scarcity.
3.  **Multi-task Pretraining**: Combines hierarchical classification loss with reconstruction and rotation prediction losses for robust feature learning.
4.  **Active Learning Pipeline**: Implements Monte Carlo (MC) Dropout for Bayesian uncertainty estimation to select the most informative unlabeled samples (`ACTIVE_TOP_K = 200`) from core datasets (MD022508, MD972138) for expert annotation.
5.  **Contrastive Learning**: Uses a supervised contrastive loss (`LAMBDA_METRIC = 0.1`) to improve the separation of species embeddings in the feature space.

## 📊 Datasets

The code automatically downloads and processes three foraminifera image datasets from Zenodo:
1.  **Endless Forams Training Set**: Primary labeled dataset for fine-tuning and evaluation.
2.  **MD022508 & MD972138 Training Sets**: Unlabeled core data used as the pool for active learning.

## 📁 Project Structure & Usage

The project is contained within a single, comprehensive Python file for ease of execution in environments like Google Colab.

### **Primary File**
- `dfh_vit_foraminifera.py`: The complete implementation, including configuration (`CFG` class), model definition (`DFHViT`), datasets, training loops, evaluation, and visualization.

### **Quick Start**
1.  **Upload to Colab**: Upload the `.py` file to a Google Colab notebook.
2.  **Install Dependencies**: The script installs necessary packages (`timm`, `torchmetrics`, `albumentations`, etc.).
3.  **Run the Script**: Execute all cells. The script will:
    - Download and extract the datasets.
    - Discover the number of fine-grained species classes from the Endless Forams directory.
    - Pretrain the model on synthetic fractal data.
    - Fine-tune the model on the real Endless Forams dataset.
    - Evaluate the model and save the best weights (`dfh_vit_forams_best.pth`).
    - Perform one round of active learning to select uncertain samples from the unlabeled pool.
    - Generate evaluation metrics, confusion matrices, and prediction visualizations.

### **Key Configuration (`CFG` Class)**
Hyperparameters are centralized in the `CFG` class. Key settings include:

```python

BATCH_SIZE = 64
IMAGE_SIZE = 224
EPOCHS_PRETRAIN = 10      # Pretraining on synthetic data
EPOCHS_FINETUNE = 10      # Fine-tuning on real data
LR_PRETRAIN = 3e-4
LR_FINETUNE = 1e-4
LAMBDA_FINE_CE = 1.0      # Weight for fine-level classification loss
LAMBDA_METRIC = 0.1       # Weight for contrastive loss
MC_DROPOUT_SAMPLES = 8    # Samples for Monte Carlo uncertainty estimation
ACTIVE_TOP_K = 200        # Number of samples to select via active learning

```


## 🔬 Connection to Related Research & Project Advancement

This project is directly inspired by, and builds upon, recent groundbreaking work at the intersection of paleontology and deep learning, specifically the study **"Classifying microfossil radiolarians on fractal pre-trained vision transformers"** (Mimura et al., 2025).

While Mimura et al. demonstrated the powerful potential of **Vision Transformers (ViT)** and **Formula-Driven Supervised Learning (FDSL)** for microfossil classification, achieving significant gains over traditional CNNs, their approach has several key limitations. Our **DFH-ViT model is explicitly designed to overcome these challenges**:

| Limitation in Mimura et al. (2025) | How DFH-ViT Addresses This Limitation |
| :--- | :--- |
| **1. Flat Classification Structure**<br>The model treats all 32 classes as peers in a single, flat classification task. This does not reflect the real-world **biological hierarchy** (e.g., Order → Family → Genus → Species). | **✅ Hierarchical Classification Design**<br>DFH-ViT introduces an explicit two-level hierarchy (`coarse_label` and `fine_label`). The model first predicts a coarse morphological group, then uses that contextual information to make a finer-grained species prediction. This mirrors expert taxonomic logic and can improve accuracy. |
| **2. Passive Learning on Fixed Data**<br>The study uses a static, pre-collected dataset. It identifies that rare or ambiguous classes perform poorly but offers no active strategy to improve them beyond collecting "a larger amount of training data." | **✅ Integrated Active Learning Pipeline**<br>DFH-ViT incorporates a Monte Carlo Dropout-based **active learning** module. It automatically identifies the **most uncertain samples** from large, unlabeled core datasets. This allows experts to **strategically label only the most informative data**, drastically improving model performance for difficult classes with optimal annotation effort. |
| **3. Limited Synthetic Data Strategy**<br>The FDSL pre-training uses general fractal (ExFractal) or contour (RCDB) images. While effective, these are not **domain-specific**; they are generic mathematical patterns not designed to mimic microfossil morphology. | **✅ Domain-Aware Synthetic Pre-training**<br>Our `SyntheticFossilFractalDataset` generates images that **specifically mimic foraminiferal morphology** (e.g., spumellarian-like radial spines, nassellarian-like conical structures). This provides more relevant inductive biases for the downstream task. |
| **4. Single-Task Learning Objective**<br>The model is optimized solely for classification accuracy using a cross-entropy loss. | **✅ Multi-Task & Contrastive Learning**<br>DFH-ViT uses a **multi-task pretraining** objective combining classification, image reconstruction, and rotation prediction. During fine-tuning, it adds a **supervised contrastive loss**. This forces the model to learn more robust, generalizable feature representations where specimens of the same species are embedded closer together. |

**In summary, your DFH-ViT project moves beyond simply *applying* a Vision Transformer to microfossils. It introduces a **more biologically informed architecture**, a **smarter data strategy** using active learning, and a **richer training paradigm** with multi-task and contrastive losses.** This represents a holistic engineering approach to building a more capable, efficient, and scalable intelligent system for paleontological analysis—a core goal of the Intelligent Software Engineering course.

