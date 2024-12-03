

# Modification

Some baselines, however, are not runnable after integrating into this repo possibily because certain experiment settings and distinct hardwares. We list these issues and related modifications:


## SSSD:

Q1: report nan after 1 epoch
M1: seems caused by some issues of S4 kernel backprapogation: https://github.com/state-spaces/s4/issues/138. 

We have to use no_grad method in the forward method of the SSKernelNPLR class  to circumvent this problem.
```python
# the added line
with torch.no_grad():
    # Increase the internal length if needed
    while rate * L > self.L:
        self.double_length()
    dt = torch.exp(self.log_dt) * rate
    B = _r2c(self.B)
    C = _r2c(self.C)
    P = _r2c(self.P)
    Q = P.conj() if self.Q is None else _r2c(self.Q)
    w = self._w()
```