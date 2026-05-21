# Genetic Symbolic Regression Report

## Kutz Cylinder (vorticity)

- Test field NRMSE: 0.7743
- Stable long-horizon integration: True
- Equations:
  - da1/dt = -0.001
  - da2/dt = mul(mul(a2, 0.232), div(a1, a2))
  - da3/dt = mul(a2, mul(a2, -0.264))

## DeepXDE Cylinder (u,v)

- Test field NRMSE: 0.1045
- Stable long-horizon integration: True
- Equations:
  - da1/dt = add(a2, -0.117)
  - da2/dt = sub(mul(a1, -0.183), a1)
  - da3/dt = sub(mul(a1, a1), add(add(mul(a2, a2), mul(a2, a2)), div(add(mul(a2, a2), mul(a1, a1)), -1.621)))
