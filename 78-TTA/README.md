## Case #78 - [CV] Test-Time Augmentation (TTA)

> 🧩 Reference: [LinkedIn Post](https://www.linkedin.com/posts/backnumber19lim_computervision-segmentation-tta-activity-7369666860138622976-2NQA?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC4i7ZsBMeUAH3UpBvhusYv1qkmTlPJ4E6E)  

Test-Time Augmentation (TTA) is a technique that applies various augmentations to a single input image during inference to generate multiple prediction results, then combines them to produce more robust inference outcomes. Particularly useful for improving model generalization performance in image segmentation problems, it's especially effective when data is limited or when prediction reliability needs to be enhanced.

By applying rotations, flips, scale changes, and other transformations to the original image, the model performs predictions on various variations it may not have encountered during training, then adopts the consensus parts as the final result. To find these consensus areas, methods such as (weighted) averaging or selecting only predictions that exceed a certain threshold can be used.

Below is example code implementing the TTA technique in PyTorch. It includes various augmentations and ensemble strategies, designed to be directly applicable to real segmentation models.

✅ The biggest advantage of TTA is that it 'emphasizes only the certain parts.' It can adopt only the areas where the model is confident as the final result, and it's practical as it can improve performance without modifying existing models.