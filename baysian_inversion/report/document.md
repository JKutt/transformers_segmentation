---
# Math frontmatter:
math:
  '\Fm'   : '\mathbf{F}(\mathbf{m})'
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
  '\cm'  : '\frac{1}{\gamma}'
---


# Probabalistic Petrophysically and Geologically guided Inversion


## Abstract


RTO and Bayesian Inversion. More soon to come.


## Start with Randomize then Optimize


Based on the work of Daniel Blatter, RTO for short is represented as the similar to the deterministic inversion.

$$
\label{minimizer}
\begin{aligned}
\underset{\m}{min} \f(\m) = \half || C_d^{-\half} (\Fm - \tilde{\d}))||^{2} + \frac{\mu}{2}||L(\m - \tilde{\m})||^{2}
\end{aligned}
$$

We set $C_d^{-\half}=W_d$ and $L=\cm$:

$$
\label{minimizer2}
\begin{aligned}
\underset{\m}{min} \f(\m) = \half || W_d \Fm - W_d \tilde{\d}||^{2} + \frac{\mu}{2}||\cm \m - \cm \tilde{\m}||^{2}
\end{aligned}
$$

For the linear case $\scriptstyle \Fm = \g\m$:

$$
\label{minimizer2}
\begin{aligned}
\underset{\m}{min} \f(\m) = \half || W_d \g \m - W_d \tilde{\d}||^{2} + \frac{\mu}{2}||\cm \m - \cm \tilde{\m}||^{2}
\end{aligned}
$$

Expand the right handside:

$$
\label{minexpand1}
\begin{aligned}
\underset{\m}{min} \f(\m) = \half \left( W_d^T \g^T \m^T \g W_d \m - 2 W_d^T W_d \g \m \tilde{\d} \right) + \frac{\mu}{2} \left( \m^T \m - 2 \m^T \tilde{\m} + \tilde{\m}^2 \right)
\end{aligned}
$$

Now take the derivative:

$$
\label{minexpand1}
\begin{aligned}
\nabla \f(\m) = \half \left( 2 W_d^T \g^T \g W_d \m - 2 W_d^T W_d \g \tilde{\d} \right) + \frac{\mu}{2} \left( 2\m - 2 \tilde{\m} \right)
\end{aligned}
$$

$$
\label{minexpand1}
\begin{aligned}
\nabla \f(\m) = W_d^T \g^T \g W_d \m - W_d^T W_d \g \tilde{\d} + \frac{\mu}{2} \m - \frac{\mu}{2} \tilde{\m}
\end{aligned}
$$

Set the gradient to 0:

$$
\label{minexpand1}
\begin{aligned}
0 = W_d^T \g^T \g W_d \m - W_d^T W_d \g \tilde{\d} + \frac{\mu}{2} \m - \frac{\mu}{2} \tilde{\m}
\end{aligned}
$$

$$
\label{minexpand1}
\begin{aligned}
 W_d^T \g^T \g W_d \m + \frac{\mu}{2} \m = W_d^T W_d \g \tilde{\d} + \frac{\mu}{2} \tilde{\m}
\end{aligned}
$$

$$
\label{minexpand1}
\begin{aligned}
 \left( W_d^T \g^T \g W_d + \frac{\mu}{2} \right) \m = W_d^T W_d \g \tilde{\d} + \frac{\mu}{2} \tilde{\m}
\end{aligned}
$$

Solve for $\m$:

$$
\label{minexpand1}
\begin{aligned}
\m = \left( W_d^T \g^T \g W_d + \frac{\mu}{2} \right)^{-1} \left( W_d^T W_d \g \tilde{\d} + \frac{\mu}{2} \tilde{\m} \right)
\end{aligned}
$$