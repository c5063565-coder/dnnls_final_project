# dnnls_final_project
A deep learning project that predicts the next frame in a comic-style story sequence using both image and text inputs. Three experiments are compared using the StoryReasoning dataset
# Overview
Given four consecutive story frames (each with an image and a text description), the model predicts the fifth frame — both its image and its description.
# Dataset
StoryReasoning (daniel3303/StoryReasoning) from HuggingFace. each with:
A PIL image
A GDI-formatted text description (objects, actions, locations)
Chain-of-Thought (cot) annotations with bounding boxes per character and object
The dataset is split 80/20 into train and validation. A held-out test set is also used.
# Architecture
**Text Encoder**
Baseline: LSTM encoder — processes tokens left to right; final hidden state used as text representation.
Innovation 1: Transformer encoder-attends to all tokens simultaneously via self-attention supports explainability via attention heatmaps.
**Visual Encoder**
Baseline: 3-layer CNN trained from scratch.
Innovation 2: ResNet-18 pretrained on ImageNet-rich pre-learned visual features transferred to story frames.
Both encoders follow a dual-pathway design (content + context) whose outputs are concatenated and projected to a shared latent space.
**Text Decoder**
LSTM decoder (identical across all experiments). Initialised from the fused latent vector to generate the predicted description.
**Visual Decoder**
Transposed convolution decoder (identical across all experiments). Reconstructs the predicted frame from the latent vector

**Full Sequence Predictor**
1.Encode all 4 context frames (image + text separately).
2.Concatenate image and text embeddings per frame.
3.Pass the 4-frame sequence through a temporal GRU.
4.Apply an attention module over GRU outputs to focus on the most relevant frames.
5.Combine the GRU last state and the attention context vector.
6.Decode to predicted image (content + context paths) and predicted text.

# Experiments
Controlled via boolean switches in the config cell. Only the switches change between experiments-all hyperparameters are fixed.
      Switch 	                     Baseline	     Exp 1	        Exp 2
USE_TRANSFORMER_ENCODER 	          False	       True	          True
USE_RESNET_ENCODER	                False      	 True	          True
USE_CONTRASTIVE_ROI              	  False	       False	        True

