## Case #79 - [ML] Ensemble Learning & Joint Training

> 🧩 Reference: [LinkedIn Post](https://www.linkedin.com/posts/backnumber19lim_ai-ml-ensemble-activity-7371380654237929472-cnOf?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC4i7ZsBMeUAH3UpBvhusYv1qkmTlPJ4E6E)  

Ensemble Learning and Joint Training are methods used to improve the performance of AI/ML models. While both share the commonality of knowledge sharing across models, they differ fundamentally in training approach and implementation philosophy.

1️⃣ Ensemble Learning

Ensemble Learning trains multiple independent models separately and then combines their prediction results to derive the final output. Since each model is trained completely independently, even if one model makes an error in a specific case, the correct predictions from other models can compensate for it.

2️⃣ Joint Training

Joint Training involves a single model being optimized for multiple tasks simultaneously.
In Joint Training, related tasks share common low-level features. The features extracted by the early layers of the model are useful across different tasks, and this shared structure allows each task to gain helpful information from others.
During training, the loss functions of all tasks are calculated together, and backpropagation is performed on the combined loss.

✅ Key Differences

• Model architecture: Ensemble consists of separate, fully functional models, whereas Joint Training uses a single architecture with shared layers combined with task-specific layers.

• Computational efficiency: Joint Training is more efficient than Ensemble because it can produce outputs for all tasks with a single forward pass during inference.

The following code shows a simple example highlighting the difference between the two methods. Ensemble trains each model separately and averages the results, while Joint Training uses one model with shared layers to handle multiple tasks simultaneously.