## Case #84 - [Time Series] RevIN

> 🧩 Reference: [LinkedIn Post](https://www.linkedin.com/posts/backnumber19lim_ai-ml-timeseries-activity-7384875839924076544-Iq0H?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC4i7ZsBMeUAH3UpBvhusYv1qkmTlPJ4E6E)  

Time series models often experience a sharp performance drop when the distributions of the training and validation data differ. This issue is particularly critical in non-stationary time series, where trends, seasonality, and representative values change over time. Traditional normalization methods (e.g., Batch Normalization or Layer Normalization) mix data from different time series or variables with distinct meanings, making it difficult for models to generalize to validation data with different distributions.

RevIN (Reversible Instance Normalization), introduced at ICLR 2022, addresses this problem by proposing an instance-level normalization method. Each time series instance is normalized individually before being fed into the model, and the model’s predictions are then reverted to the original scale. This allows the model to learn distribution-invariant patterns, while the final predictions retain their meaningful original scale.

RevIN consists of two simple steps:
1️⃣ Normalization: For each input sequence x, compute its instance-specific mean μ and standard deviation σ, then transform it as (x - μ) / σ.
2️⃣ Denormalization: For the model’s prediction y, apply y × σ + μ to restore it to the original scale.

The attached image shows a simple experimental code and results demonstrating the effect of RevIN. The generate_nonstationary_data function first creates non-stationary time series whose mean and standard deviation vary for each sequence. Then, the performance of the same LSTM model is compared with and without RevIN. The validation results clearly show that applying RevIN leads to superior performance.

![RevIN Implementation and Results](https://media.licdn.com/dms/image/v2/D5622AQECv2krZE_UxA/feedshare-shrink_2048_1536/B56ZnwkbwOHAAw-/0/1760677707861?e=1762387200&v=beta&t=VCwZasTLqdRb1psnIgHiXmlugeHPvSIaKIapX_1RV5E)