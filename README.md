# Drone Video Anomaly Detection by Future Segmentation and Spatio-Temporal Relational Modeling

In traffic surveillance, accurate video anomaly detection is vital for public safety, yet environmental changes, occlusions, and visual obstructions pose significant challenges. In this research, we introduce **DAD-FSM**, an innovative drone-based video anomaly detection system that leverages a spatio-temporal relational cross-transformer to enhance the encoding of visual and temporal features for future segmentation. Additionally, we propose the motion-aware frame prediction loss function (MAFL) to improve the model's representation and the background and foreground separation of moving objects. Our method achieves state-of-the-art (SOTA) AUC scores of 68.13% on the UIT-ADrone dataset and 73.5% mAUC on the Drone-Anomaly dataset, surpassing previous methods by 2.68% and 5.71% respectively. These results demonstrate the potential of our model for broad use in traffic surveillance applications.

![DADFSM](/figures/Overallmodel_alt.png)

## Model Performance on Drone-Anomaly and UIT-ADrone datasets
### Table 1: Comparison of methods on different scenes of the Drone-Anomaly dataset. AUC (higher is better) and EER (lower is better) scores are reported.

```markdown
| Method   | Railway      |             | Highway      |             | Bike roundabout |             | Vehicle roundabout |             | Crossroads   |             |
|----------|---------------|-------------|--------------|-------------|-----------------|-------------|--------------------|-------------|--------------|-------------|
|          | AUC ↑        | EER ↓       | AUC ↑        | EER ↓       | AUC ↑           | EER ↓       | AUC ↑              | EER ↓       | AUC ↑        | EER ↓       |
| DAD-FSM  | 76.01        | 0.30        | 87.08        | 0.20        | 74.00           | 0.29        | 80.66              | 0.30        | 49.09        | 0.50        |
```

### Table 2: Comparison of methods on two major drone anomaly datasets, UIT-ADrone and Drone-Anomaly. AUC and EER scores are reported. For the Drone-Anomaly dataset, mAUC and mEER scores are calculated over the five major scenes.

```markdown
| Method   | UIT-ADrone    |            | Drone-Anomaly  |            |
|----------|---------------|------------|----------------|------------|
|          | AUC ↑         | EER ↓      | mAUC ↑         | mEER ↓     |
| DAD-FSM  | 68.13         | 0.34       | 73.51          | 0.30       |
```
