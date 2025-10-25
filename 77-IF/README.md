## Case #77 - [ML] Instance-level Explainability with Influence Functions

> 🧩 Reference: [LinkedIn Post](https://www.linkedin.com/posts/backnumber19lim_explainability-ai-ml-activity-7368462677364412416-S2CD?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC4i7ZsBMeUAH3UpBvhusYv1qkmTlPJ4E6E)  

Influence Functions is a powerful technique that explains AI/ML model predictions from the perspective of "which training data had the biggest impact on this prediction?" Proposed in the paper "Understanding Black-box Predictions via Influence Functions", this method quantitatively calculates the influence of each training data point by tracing the model's predictions back through the learning process. 

Removing training data points one by one and retraining the model is prohibitively slow. Instead, the authors infinitesimally upweight the data point in the loss function to achieve a similar effect as data removal. This allows us to compute the influence of a specific training data point using the Hessian matrix and gradients of the loss function. 

This instance-level approach is fundamentally different from feature-level methods. For instance, SHAP (Shapley Additive Explanations) calculates how each variable (feature) affects predictions, whereas Influence Functions calculates how each training data point affects predictions. In other words, SHAP answers "which features were important for the prediction?" while Influence Functions answers "which training data points shaped the prediction?" 

Below is a simple example of Influence Functions implemented in PyTorch. Using the PyTorch implementation of Influence Functions, this code calculates the influential training samples for specific test samples on the MNIST dataset.