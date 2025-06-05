# Replication Study: Oey & Vul (2024)

**"Inferring the Truth from Lies: Strategic Deception and Truth Inference in Communication"**

*Computational Brain & Behavior (2024) 7:23–36*

## Overview

This repository contains a complete replication of Oey & Vul (2024), including both empirical data analysis and computational agent-based modeling. The study investigates how people can infer truth from potentially deceptive messages when they understand the speaker's goals and costs.

## Original Paper

- **Authors**: Lauren Oey, Edward Vul
- **Journal**: Computational Brain & Behavior (2024)
- **DOI**: [10.1007/s42113-023-00187-0](https://doi.org/10.1007/s42113-023-00187-0)

## Replication Structure

### Part 1: Human Experiment Analysis
- **Figure 2**: Bias distributions showing strategic lying and correction
- **Truth recovery analysis**: Scatter plots showing judge accuracy
- **Individual correlation analysis**: Key negative correlation finding
- **Statistical validation**: All significance tests and effect sizes

### Part 2: Computational Simulations  
- **Level-k reasoning agents** (L0, L1, L2, L3, L4)
- **Utility functions** for deceptive and cooperative senders
- **Bayesian inference** for judges correcting bias
- **Parameter sensitivity analysis** 
- **Population dynamics** with mixed agent types

## Installation and Setup

```bash
# Clone the repository
git clone https://github.com/batuhansolmaz/InferringTruth-Replication-Study.git
cd InferringTruth-Replication-Study

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run complete replication
python run_complete_replication.py
```

## Files

### Core Scripts
- `comprehensive_analysis.py` - Human experiment data analysis
- `computational_simulations.py` - Agent-based modeling framework
- `run_complete_replication.py` - Master script running full replication

### Data Files
- `sender.csv` - Sender behavior data (204 participants)
- `receiver.csv` - Receiver/judge behavior data
- `trials.csv` - Trial-level data
- `raw.csv` - Original raw experimental data

### Generated Visualizations

**Human Experiment Analysis:**
- `figure2_replication.png` - Core bias distributions
- `truth_recovery_analysis.png` - Truth recovery scatter plots  
- `individual_sender_judge_correlation.png` - Individual correlation analysis

**Computational Simulations:**
- `level_k_evolution_grid.png` - Level-k reasoning evolution
- `enhanced_parameter_sensitivity.png` - Parameter sensitivity analysis
- `population_dynamics.png` - Population dynamics

## Key Findings Replicated

### Empirical Results ✅
- **Strategic lying**: Senders bias messages toward their assigned goals
- **Cost effects**: Linear cost → larger biases than quadratic cost
- **Strategic correction**: Judges correct bias in opposite direction
- **Individual correlation**: Strong negative relationship (r ≈ -0.71)
- **Truth recovery**: High accuracy despite systematic deception
- **Statistical significance**: All effects p < 0.0001

### Theoretical Predictions ✅
- **Equilibrium convergence**: Level-k reasoning reaches stable state
- **Bias patterns**: Deceptive senders show systematic directional bias
- **Bayesian correction**: Judges use optimal inference to counter bias
- **Parameter control**: Cost parameter m controls bias-accuracy tradeoff

## Technical Implementation

### Data Analysis
- **Language**: Python (pandas, matplotlib, scipy)
- **Sample**: 204 participants, ~20,000 trials
- **Design**: 2×2 (cost: linear/quadratic × goal: over/under estimation)

### Computational Modeling
- **Framework**: Level-k reasoning with recursive strategic thinking
- **Decision rule**: Softmax with temperature α=40
- **State space**: 21 discrete values in [0,1] range
- **Utilities**: Implemented from equations (1)-(3) in paper

### Agent Framework
```python
# Utility functions
U_J(k, kest) = -(kest - k)²                    # Judge utility (Eq. 1)
U_SD(k, ksay, kest, m) = (kest - k) - m(ksay - k)²  # Deceptive sender (Eq. 2)  
U_SC(k, kest) = -(kest - k)²                   # Cooperative sender (Eq. 3)

# Decision rules  
P_S(ksay|k) ∝ exp(α × E[U_S])                 # Sender decision (Eq. 4)
P_J(kest|ksay) ∝ exp(α × E[U_J])              # Judge decision (Eq. 5)
```

### Level-k Reasoning
1. **L0 Judge**: Literal interpretation (kest = ksay)
2. **L1 Sender**: Responds optimally to L0 judge  
3. **L1 Judge**: Bayesian inference about L1 sender
4. **L2 Sender**: Responds optimally to L1 judge
5. **Continues** until convergence (~L4-L5)

## Validation

All key statistics from original paper successfully replicated:
- Mean biases by condition (within 0.01 units)
- Significance levels (all p < 0.0001)  
- Individual correlation coefficient (r ≈ -0.71)
- Truth recovery R² values across conditions
- Level-k reasoning convergence patterns
- Parameter sensitivity behavior

## Repository Structure

```
InferringTruth-Replication-Study/
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git exclusions
├── comprehensive_analysis.py               # Human experiment analysis
├── computational_simulations.py            # Agent-based modeling
├── run_complete_replication.py             # Master replication script
├── sender.csv                             # Sender behavior data
├── receiver.csv                           # Receiver behavior data  
├── trials.csv                             # Trial-level data
├── raw.csv                                # Raw experimental data
└── [Generated visualizations]              # PNG output files
```

## Citation

```bibtex
@article{oey2024inferring,
  title={Inferring the Truth from Lies: Strategic Deception and Truth Inference in Communication},
  author={Oey, Lauren and Vul, Edward},
  journal={Computational Brain \& Behavior},
  volume={7},
  pages={23--36},
  year={2024},
  publisher={Springer}
}
```
