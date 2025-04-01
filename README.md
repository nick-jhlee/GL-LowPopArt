# Captions for Figures of ICML 2025 Rebuttal

## Fig1.png
The arm-set (of 150 arms) is initially sampled from the unit Frobenius ball and fixed throughout the experiments.
Each experiment first constructs a random instance of symmetric rank-1 $3 \times 3$ matrix via $u u^T$ where $u = \frac{v}{\lVert v \rVert}$ and $v \sim \mathcal{N}_3(0, 1)$.
We repeat the experiments 10 times for statistical significance.
For nuclear norm regularized estimator (Stage I) and GL-LowPopArt (Stage I + II), we use the precise theoretical hyperparameter without further tuning.
For Burer-Monteiro factorization (BMF), we use the initialization of $U_0$ with each entry sampled i.i.d. from $10^{-3} \cdot \mathcal{N}(0, 1)$ to simulate small initialization (Stoger & Soltanolkotabi (2021); Chung & Kim (2023)).
We use the learning rate of $0.3$ and either stop when the gradient norm is below $10^{-6}$, or when the maximum number of iterations of $10^4$ is reached.
The results (Fig1.png) show that the matrix Catoni estimator outperforms the other methods in the nuclear norm error across the considered sample sizes.

## Fig2.png
Under the same setting, we compare the performance of the matrix Catoni estimator with different initial(pilot) estimators. zero, random, and MLE obtained from $N = 3 \cdot 10^4$ samples. The results (Fig2.png) show that one needs to use a good initial estimator to achieve the best performance with the matrix Catoni.