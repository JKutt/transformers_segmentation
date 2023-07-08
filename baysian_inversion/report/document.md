---
# Math frontmatter:
math:
  '\Fm' : '\mathbf{F}(\mathbf{m})'
  '\d'    : '\mathbf{d}'
  '\pfx'  : '\mathbf{P}_{x}'
  '\pfy'  : '\mathbf{P}_{y}'
  '\pfz'  : '\mathbf{P}_{z}'
  '\l'    : '\mathbf{L}'
  '\r'    : '\mathbf{r}'
  '\m'    : '\mathbf{m}'
  '\f'    : '\mathcal{f}'
  '\half' : '\frac{1}{2}'
  '\g'    : '\mathbf{G}'
---


# RTO and Bayesian Inversion


## Abstract


Soon come.


## Randomize then Optimize


Based on the work of Daniel Blatter, RTO for short is represented as the similar to the deterministic inversion.

$$
\label{minimizer}
\begin{aligned}
\underset{\m}{min} \f(\m) = \half || C_d^{-\half} (\Fm - \tilde{\d}))||^{2} + \frac{\mu}{2}||L(\m - \tilde{\m})||^{2}
\end{aligned}
$$

$$
\label{minexpand}
\begin{aligned}
\underset{\m}{min} \f(\m) = \half C_d^{-1} (\Fm - \tilde{\d})^{T} (\Fm - \tilde{\d}) + \frac{\mu}{2} L^{2}(\m - \tilde{\m})^{T}(\m - \tilde{\m})
\end{aligned}
$$

for linear case $\scriptstyle \Fm = \g\m$:

$$
\label{minexpand1}
\begin{aligned}
\underset{\m}{min} \f(\m) = \half C_d^{-1} (\g \m - \tilde{\d})^{T} (\g\m - \tilde{\d})) + \frac{\mu}{2} L^{2}(\m - \tilde{\m})^{T}(\m - \tilde{\m})
\end{aligned}
$$

$$
\label{minexpand2}
\begin{aligned}
\underset{\m}{min} \hspace{0.125cm} \f(\m) = \half C_d^{-1} (\m^T \g^T \g \m - 2\m^T \g^T \tilde{\d} + \tilde{\d}^T \tilde{\d}) - \frac{\mu}{2} L^2 (\m^T \m -2 \tilde{\m}^T \m + \tilde{\m}^T \tilde{\m})
\end{aligned}
$$

Take gradient:

$$
\label{minexpand3}
\begin{aligned}
\nabla \f(\m) = \half C_d^{-1} (\g^T \g \m - 2\g^T \tilde{\d}) - \frac{\mu}{2} L^2 (\m -2 \tilde{\m}^T )
\end{aligned}
$$

Set gradient to 0 and solve of $\m$:

$$
\label{minexpand4}
\begin{aligned}
0 = \half C_d^{-1} \g^T \g \m - C_d^{-1} \g^T \tilde{\d} - \frac{\mu}{2} L^2 \m - \mu L^2 \tilde{\m}
\end{aligned}
$$

$$
\label{minexpand5}
\begin{aligned}
\half C_d^{-1} \g^T \g \m - \frac{\mu}{2} L^2 \m = C_d^{-1} \g^T \tilde{\d} + \mu L^2 \tilde{\m}
\end{aligned}
$$

$$
\label{minexpand6}
\begin{aligned}
(\half C_d^{-1} \g^T \g - \frac{\mu}{2} L^2) \m = C_d^{-1} \g^T \tilde{\d} + \mu L^2 \tilde{\m}
\end{aligned}
$$

$$
\label{minexpand7}
\begin{aligned}
 \m = (\half C_d^{-1} \g^T \g - \frac{\mu}{2} L^2)^{-1} (C_d^{-1} \g^T \tilde{\d} + \mu L^2 \tilde{\m})
\end{aligned}
$$