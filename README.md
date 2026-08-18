WDcVHPPI
Wide & Deep Context-aware Virus–Human Protein–Protein Interaction Prediction

WDcVHPPI is a computational framework for predicting human–virus protein–protein interactions (PPIs) by integrating protein-level, network-level, and viral taxonomic information within a Wide & Deep learning architecture.

The framework is designed to capture both direct feature associations and complex nonlinear relationships between viral and human proteins.

Overview

Virus–host protein–protein interactions play an important role in viral infection, replication, host manipulation, and immune evasion. Experimental identification of these interactions at large scale is challenging and computational approaches can help prioritize potential interactions for further investigation.

WDcVHPPI integrates multiple sources of biological information to provide a context-aware approach for human–virus PPI prediction.

Main Components
Protein-level representations
Human protein–protein interaction network information
Viral taxonomic information
Viral family and order context
Wide & Deep learning architecture
Human–virus PPI prediction
Framework
Protein Features
       │
       ├──────────────┐
       │              │
       ▼              ▼
Human PPI        Viral Taxonomy
Network          Information
       │              │
       └───────┬──────┘
               ▼
       Feature Integration
               │
        ┌──────┴──────┐
        ▼             ▼
   Wide Component  Deep Component
        │             │
        └──────┬──────┘
               ▼
        Integrated Model
               │
               ▼
      Human–Virus PPI Prediction
Key Features
Protein Representation

Protein-level features are used to represent the molecular characteristics of viral and human proteins and provide the fundamental input for interaction prediction.

Human PPI Network

Human protein–protein interaction information provides biological context by representing relationships between host proteins within the cellular interaction network.

Viral Taxonomic Context

Viral taxonomy is incorporated to provide additional biological context for different viral groups. The framework can consider viral family and order information when modeling virus–host interactions.

Wide & Deep Architecture

WDcVHPPI combines two complementary learning components.

The Wide component captures informative feature combinations and direct associations, while the Deep component learns nonlinear and higher-order relationships within the integrated feature space.

Wide Learning
     +
Deep Learning
     ↓
Integrated Representation
     ↓
PPI Prediction
Prediction Task

Given a viral protein and a human protein, WDcVHPPI estimates the probability that the two proteins interact.

Viral Protein
      +
Human Protein
      +
Biological Context
      ↓
   WDcVHPPI
      ↓
Interaction Probability

The predicted interaction scores can be used to prioritize candidate virus–host interactions for downstream analysis and experimental validation.

Evaluation

WDcVHPPI supports evaluation using commonly used metrics for binary interaction prediction, including:

AUPRC
AUC
Accuracy

These complementary metrics provide different perspectives on model discrimination and classification performance.

Applications

WDcVHPPI can be applied to:

Human–virus PPI prediction
Host protein prioritization
Virus–host interaction network analysis
Viral infection mechanism studies
Viral host-dependency analysis
Investigation of virus–host biological relationships
Candidate interaction prioritization
Computational drug-discovery research
General Workflow
Data Collection
      ↓
Data Preprocessing
      ↓
Protein Feature Extraction
      ↓
Human PPI Network Construction
      ↓
Viral Taxonomic Context
      ↓
Feature Integration
      ↓
Wide & Deep Learning
      ↓
PPI Prediction
      ↓
Candidate Interaction Prioritization

WDcVHPPI is designed as a reproducible computational framework for virus–host PPI prediction.

For reproducible experiments, the same dataset, preprocessing procedure, feature representation, model configuration, and evaluation protocol should be maintained.

Predicted interactions represent computationally inferred candidates and should be independently validated through appropriate experimental approaches.
