# Drone Video Anomaly Detection by Future Segmentation and Spatio-Temporal Relational Modeling

In traffic surveillance, accurate video anomaly detection is vital for public safety, yet environmental changes, occlusions, and visual obstructions pose significant challenges. In this research, we introduce **DAD-FSM**, an innovative drone-based video anomaly detection system that leverages a spatio-temporal relational cross-transformer to enhance the encoding of visual and temporal features for future segmentation. Additionally, we propose the motion-aware frame prediction loss function (MAFL) to improve the model's representation and the background and foreground separation of moving objects. Our method achieves state-of-the-art (SOTA) AUC scores of 68.13% on the UIT-ADrone dataset and 73.5% mAUC on the Drone-Anomaly dataset, surpassing previous methods by 2.68% and 5.71% respectively. These results demonstrate the potential of our model for broad use in traffic surveillance applications.

![DADFSM](/figures/Overallmodel_alt.png)

### Table 2: Comparison of methods on two major drone anomaly datasets, UIT-ADrone and Drone-Anomaly.

| Method   | UIT-ADrone |      | Drone-Anomaly |      |
|----------|------------|------|---------------|------|
|          | AUC ↑      | EER ↓| mAUC ↑        | mEER ↓|
| FFP      | 53.56      | 0.47 | 57.94         | 0.43 |
| STD      | 57.05      | 0.45 | 52.64         | 0.47 |
| MNAD     | 55.88      | 0.46 | 52.34         | 0.51 |
| MLEP     | 51.55      | 0.47 | 55.00         | 0.48 |
| ANDT     | 60.50      | 0.42 | 63.05         | 0.43 |
| ASTT     | 65.45      | 0.39 | 67.80         | 0.36 |
| DAD-FSM  | 68.13      | 0.34 | 73.51         | 0.30 |
