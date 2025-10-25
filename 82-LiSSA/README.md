## Case #82 - [ML] Breaking the O(n^3) Barrier: Stochastic Approach for Matrix Inversion

> 🧩 Reference: [LinkedIn Post](https://www.linkedin.com/posts/backnumber19lim_ai-ml-matrix-activity-7376537095869407232-Q1BX?utm_source=share&utm_medium=member_desktop&rcm=ACoAAC4i7ZsBMeUAH3UpBvhusYv1qkmTlPJ4E6E)  

Matrix inversion operations are an unavoidable bottleneck in large-scale AI/ML models. Traditional methods like Gaussian-Jordan elimination scale cubically with the number of parameters n, making them impractical for current AI/ML models with tens of millions to billions of parameters. This problem becomes even more severe in second-order optimization requiring Hessian matrix inversion or influence function calculations (see Case #77). LiSSA (Linear-time Stochastic Second-Order Algorithm) is one approach that theoretically achieves O(n) time complexity.

The core of LiSSA lies in the stochastic approximation of Neumann series. It approximates the Neumann series (I - H)^(-1) = I + H + H^2 + ... through stochastic sampling. The key insight is that instead of directly computing the entire n×n Hessian matrix, it samples only a portion of data at each iteration and computes only Hessian-vector products. This reduces memory usage to O(n) and makes computation time scale linearly with the number of iterations.

It iteratively updates using X[i, j] = ∇f + (I - H)X[i, j-1] with Hessians from randomly selected samples across the entire dataset. This approach enables efficient approximation of the product of Hessian inverse and gradient without directly computing or storing the entire Hessian matrix.

To demonstrate LiSSA's core concepts, we performed a simple benchmark using conceptual code based on the official implementation. As shown in the second image, we confirmed approximately 1.6x-2x speedup compared to Gaussian-Jordan elimination.

⚠️ The code example below is an implementation for demonstrating LiSSA concepts. In actual development, using highly optimized numpy.linalg.inv() is more efficient. LiSSA is primarily applicable in special situations like deep learning model Hessians where the entire matrix cannot be loaded into memory.

![Results](https://media.licdn.com/dms/image/v2/D5622AQF7HYAW1qyGOw/feedshare-shrink_800/B56Zl65Z4sIsAk-/0/1758703492866?e=1762387200&v=beta&t=uKUXWVu5-b5MRWeogSMn26MahHHZskTIIJVR-fCCdU0)