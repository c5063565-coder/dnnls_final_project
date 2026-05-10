Multimodal Story Sequence Prediction
A deep learning project that predicts the next frame in a comic-style story sequence using both image and text inputs. Three experiments are compared using the StoryReasoning dataset.
---
Overview
Given four consecutive story frames (each with an image and a text description), the model predicts the fifth frame — both its image and its description.
---
Dataset
StoryReasoning (`daniel3303/StoryReasoning`) from HuggingFace. Each story contains five frames, each with:
A PIL image
A GDI-formatted text description (objects, actions, locations)
Chain-of-Thought (CoT) annotations with bounding boxes per character and object
The dataset is split 80/20 into train and validation. A held-out test set is also used.
---
Architecture
Text Encoder
Baseline: LSTM encoder — processes tokens left to right; final hidden state used as text representation.
Innovation 1: Transformer encoder — attends to all tokens simultaneously via self-attention; supports explainability via attention heatmaps.
Visual Encoder
Baseline: 3-layer CNN trained from scratch.
Innovation 2: ResNet-18 pretrained on ImageNet — rich pre-learned visual features transferred to story frames.
Both encoders follow a dual-pathway design (content + context) whose outputs are concatenated and projected to a shared latent space.
Text Decoder
LSTM decoder (identical across all experiments). Initialised from the fused latent vector to generate the predicted description.
Visual Decoder
Transposed convolution decoder (identical across all experiments). Reconstructs the predicted frame from the latent vector.
Full Sequence Predictor
Encode all 4 context frames (image + text separately).
Concatenate image and text embeddings per frame.
Pass the 4-frame sequence through a temporal GRU.
Apply an attention module over GRU outputs to focus on the most relevant frames.
Combine the GRU last state and the attention context vector.
Decode to predicted image (content + context paths) and predicted text.
---
Experiments
Controlled via boolean switches in the config cell. Only the switches change between experiments — all hyperparameters are fixed.
Switch	Baseline	Exp 1	Exp 2
`USE\_TRANSFORMER\_ENCODER`	False	True	True
`USE\_RESNET\_ENCODER`	False	True	True
`USE\_CONTRASTIVE\_ROI`	False	False	True
Experiment 2 adds contrastive grounding losses:
ReID loss: crops of the same character/object from different frames should have similar embeddings (MSE).
InfoNCE contrastive loss: ROI embeddings are aligned with their corresponding frame text embeddings using temperature-scaled cross-entropy.
---
Hyperparameters
Parameter	Value
Embedding dim	16
Latent dim	16
LSTM layers	1
Dropout	0.1
Training epochs	15
Batch size	8
Learning rate	0.001
Max token length	120
Image size	60 × 125
Contrastive temperature (τ)	0.07
---
Training Pipeline
Cell 12 — External Text Pretraining
Transformer only: Stage A on WikiText-2 (general English structure), then Stage B on TinyStories (short character-action stories — closest freely available domain to comic descriptions).
LSTM: Stage B on TinyStories only (already has provided pretrained weights; fine-tuned gently).
Cell 13 — StoryReasoning Text Fine-tuning
Both encoders are fine-tuned on StoryReasoning story descriptions. Best checkpoint is selected by validation loss. Text encoder is then frozen.
Cell 14 — Visual Encoder Pretraining
Stage A: STL-10 (5,000 images, 10 categories — similar visual diversity to story frames).
Stage B: StoryReasoning fine-tuning with dual loss (content reconstruction + context consistency).
ResNet uses a lower learning rate than CNN to preserve ImageNet weights.
Cell 15 — Main Training
All parameters unfrozen. Combined loss:
Image prediction: L1 loss
Context consistency: MSE loss
Text prediction: Cross-entropy (with teacher forcing)
ReID + contrastive losses (Experiment 2 only, weighted by λ)
---
Losses
Loss	Weight	Active in
Image L1	1.0	All
Context MSE	1.0	All
Text cross-entropy	1.0	All
ReID MSE	0.10	Exp 2
InfoNCE contrastive	0.10	Exp 2
---
Explainability (Cell 16)
Transformer attention heatmaps: Layer 1 self-attention weights visualised as a token × token matrix, showing which tokens the model attends to.
Grad-CAM: Gradient-weighted class activation maps overlaid on input frames, showing which image regions most influenced the ResNet visual encoding.
Prediction visualisation: Side-by-side of input frame, ground truth frame, and predicted frame, plus decoded predicted text vs ground truth description.
---
Results
Fill in after running all three experiments.
Experiment	Text PT Loss	Visual PT Loss	Main Val Loss	Improvement vs Baseline
Baseline (LSTM + CNN)	???	???	???	—
Exp 1 (Transformer + ResNet)	???	???	???	???%
Exp 2 (+ Contrastive)	???	???	???	???%
Ablation: Transformer Encoder Depth
Configuration	Text Val Loss	Main Val Loss	Training Time
LSTM Baseline	???	???	???
Transformer 2-layer	???	???	???
Transformer 4-layer	???	???	???
To run the 4-layer ablation: set `USE\_TRANSFORMER\_ENCODER = True` in Cell 3, change `num\_layers=4` in `EncoderTransformer` in Cell 11, then re-run Cells 11, 12, 13, and 15.
---
File Structure
```
DNN/DL\_Checkpoints/
├── checkpoints/          # Model checkpoints (.pth)
└── experiments/
    ├── baseline/         # Logs and figures for baseline
    ├── exp1\_transformer\_resnet/
    └── exp2\_contrastive/
```
---
Requirements
PyTorch
torchvision
transformers (BertTokenizer)
datasets (HuggingFace)
BeautifulSoup4
matplotlib, numpy
Google Colab (for Drive mounting and GPU)
---
How to Run
Open the notebook in Google Colab.
Mount Google Drive (Cell 2).
Set experiment switches in Cell 3.
Run cells sequentially: 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16.
After all three experiments, fill in the results table in Cell 17 and generate the comparison bar chart.
To switch experiments, return to Cell 3, change the switches, and re-run from Cell 11 onward.
