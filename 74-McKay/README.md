## Case #74 - [ML] Is Your Metric Really Stable?

> 🧩 Reference: [LinkedIn Post](https://www.linkedin.com/posts/backnumber19lim_ai-ml-cv-activity-7355522071948353536-Xw0u?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC4i7ZsBMeUAH3UpBvhusYv1qkmTlPJ4E6E)  

When seeking confidence in the stability of ML model performance, we often conduct repeated experiments to measure performance multiple times. With these multiple performance measurements, we calculate the mean, standard deviation, and the coefficient of variation (CV) - the ratio of standard deviation to mean. CV is widely used because it allows comparison of variability across data with different scales.

However, the CV calculated here is merely a point estimate. For example, if we have a criterion of CV < 0.05 and our calculated CV is 0.04, we cannot definitively say the criterion is met. If the confidence interval is [0.01, 0.07], there's a possibility of not meeting the criterion. Therefore, as with all estimators, confidence interval calculation is necessary, and the confidence interval for CV can be calculated using the method proposed by McKay in 1932.

The McKay method calculates CV confidence intervals using the chi-square distribution. Since visualizing this through code is more intuitive than explaining with formulas, I've included the actual implementation function in the attached image. However, several conditions are required to use this method:

1️⃣ CV should be less than 0.33 (recommended)
2️⃣ Sample size should be 10 or more
3️⃣ Normal distribution assumption
4️⃣ Positive data values

Looking at the execution results in the attached image, you can see how much different insights this provides for the same data. When measuring model performance 10 times, sharing results as "CV of 1.73% showing stable performance, with 95% confidence interval upper bound around 3%, ensuring consistency" conveys model reliability more effectively than basic descriptive statistics like "mean 86%, standard deviation 0.015"

![Results](https://media.licdn.com/dms/image/v2/D5622AQE4KxxyKfjNCA/feedshare-shrink_2048_1536/B56ZhQQWgLHMAo-/0/1753693120726?e=1762992000&v=beta&t=pGvawtKmhaxnf1qHrIlY907YxxGfjdPrnb-_MJI5m4U)