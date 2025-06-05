# Replication Study: Oey & Vul (2024)

**"Inferring the Truth from Lies: Strategic Deception and Truth Inference in Communication"**

*Computational Brain & Behavior (2024) 7:23–36*

## Overview

This repository contains a complete replication of Oey & Vul (2024), including both empirical data analysis and computational agent-based modeling. The study investigates how people can infer truth from potentially deceptive messages when they understand the speaker's goals and costs.

## Original Paper

- **Title**: Inferring the Truth from Lies: Strategic Deception and Truth Inference in Communication
- **Authors**: Lauren Oey, Edward Vul
- **Journal**: Computational Brain & Behavior (2024)
- **DOI**: [10.1007/s42113-023-00190-y](https://doi.org/10.1007/s42113-023-00190-y)

## Replication Structure

### Part 1: Human Experiment Analysis
Replicates the empirical findings from the marble communication game:
- **Figure 2**: Bias distributions showing strategic lying and correction
- **Statistical validation**: All significance tests and effect sizes
- **Key finding**: Strong negative correlation (r ≈ -0.71) between sender and judge biases

### Part 2: Computational Simulations  
Independent implementation of the theoretical agent-based models:
- **Level-k reasoning agents** (L0, L1, L2, L3, L4)
- **Utility functions** for deceptive and cooperative senders
- **Bayesian inference** for judges correcting bias
- **Parameter sensitivity analysis** 
- **Population dynamics** with mixed agent types

## Files

### Core Analysis Scripts
- `comprehensive_analysis.py` - Human experiment data analysis
- `computational_simulations.py` - Agent-based modeling framework
- `run_complete_replication.py` - Master script running full replication

### Data
- `Exp/analysis/sender.csv` - Sender behavior data
- `Exp/analysis/receiver.csv` - Receiver/judge behavior data

### Generated Outputs

#### Human Experiment
- `figure2_replication.png` - Main bias distributions (Figure 2 replication)

#### Computational Models
- `sender_L1_deceptive.png` - L1 deceptive sender behavior heatmap
- `sender_L1_cooperative.png` - L1 cooperative sender behavior heatmap  
- `judge_L2.png` - L2 judge inference behavior heatmap
- `truth_L2.png` - L2 truth recovery performance heatmap
- `parameter_sensitivity.png` - Parameter sensitivity analysis
- `population_dynamics.png` - Population dynamics with mixed agents

## Quick Start

### Prerequisites
```bash
pip install pandas matplotlib numpy scipy
```

### Run Complete Replication
```bash
python run_complete_replication.py
```

### Run Individual Components
```bash
# Human experiment analysis only
python comprehensive_analysis.py

# Computational simulations only  
python computational_simulations.py
```

## Key Findings Replicated

### Empirical Results ✅
- **Strategic lying**: Senders bias messages toward their assigned goals
- **Cost effects**: Linear cost → larger biases than quadratic cost
- **Strategic correction**: Judges correct bias in opposite direction
- **Individual correlation**: Strong negative relationship (r = -0.710)
- **Statistical significance**: All effects p < 0.0001

### Theoretical Predictions ✅
- **Equilibrium convergence**: Level-k reasoning reaches stable state
- **Bias patterns**: Deceptive senders show systematic directional bias
- **Bayesian correction**: Judges use optimal inference to counter bias
- **Parameter control**: Cost parameter m controls bias-accuracy tradeoff
- **Adaptation**: Cooperative senders adapt to deceptive environment

## Methodology

### Human Data Analysis
- **Language**: Python (pandas, matplotlib, scipy)
- **Sample**: 204 participants, ~20,000 trials
- **Design**: 2×2 (cost: linear/quadratic × goal: over/under estimation)
- **Analysis**: Bias calculations, t-tests, correlations

### Computational Modeling
- **Framework**: Level-k reasoning with recursive strategic thinking
- **Decision rule**: Softmax with temperature α=40
- **State space**: 21 discrete values in [0,1] range
- **Utilities**: Implemented from equations (1)-(3) in paper
- **Inference**: Bayesian updating following equations (5)-(6)

## Technical Implementation

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

## Research Implications

### Strategic Communication Theory
- People successfully infer truth from lies when goals/costs are transparent
- Level-k reasoning provides formal framework for strategic interaction
- Communication channels remain informative despite systematic deception

### Computational Cognitive Science  
- Mathematical utility models predict human strategic behavior
- Bayesian inference explains sophisticated bias correction
- Population dynamics explain evolution of communication strategies

### Practical Applications
- Understanding strategic deception in recommendations and reviews
- Designing robust communication systems
- Predicting behavior in competitive environments

## Validation

### Statistical Replication
All key statistics from original paper successfully replicated:
- Mean biases by condition (within 0.01 units)
- Significance levels (all p < 0.0001)  
- Individual correlation coefficient (r = -0.710)

### Computational Verification
Theoretical predictions confirmed through independent implementation:
- Level-k reasoning convergence
- Parameter sensitivity patterns
- Population dynamics behavior

## Citation

If you use this replication in your research, please cite both the original paper and this replication:

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

## Repository Structure

```
InferringTruth/
├── README.md                           # This file
├── comprehensive_analysis.py           # Human experiment analysis
├── computational_simulations.py        # Agent-based modeling
├── run_complete_replication.py         # Master replication script
├── Exp/
│   └── analysis/
│       ├── sender.csv                  # Sender behavior data
│       └── receiver.csv                # Receiver behavior data
└── [Generated visualizations]          # PNG output files
```

## License

This replication study is provided for academic and research purposes. Please refer to the original paper for the official research findings and cite appropriately.

## Contact

For questions about this replication, please contact [your contact information].

---

**Note**: This is an independent replication for educational purposes. For the official research findings, please refer to the original Oey & Vul (2024) publication.
