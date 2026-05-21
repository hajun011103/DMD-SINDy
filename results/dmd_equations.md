# DMD+SINDy Equation Report

## Kutz Cylinder (vorticity)

- Hyperparameters: rank=9, library=meanfield_poly2, diff=finite_difference, lambda=0.1
- Test field NRMSE: 0.2120; active terms: 9; shift correlation: 0.088
- Reference shift-mode alignment: 1.000
- Equations:
  - da1/dt = -0.0007 a1 -0.2033 a2 -0.0048 a1 a3
  - da2/dt = 0.2061 a1 +0.0003 a2 +0.0003 a2 a3
  - da3/dt = 0.0650 a3 +0.2893 a1^2 -0.2900 a2^2

## DeepXDE Cylinder (u,v)

- Hyperparameters: rank=10, library=poly2, diff=finite_difference, lambda=0.215
- Test field NRMSE: 0.0877; active terms: 17; shift correlation: 0.889
- Equations:
  - da1/dt = -0.1455 +0.0594 a1 +1.0464 a2 +0.0248 a1^2 +0.0194 a2^2
  - da2/dt = -1.1089 a1 -0.0546 a2 -0.0401 a1^2 -0.0308 a2^2
  - da3/dt = 0.6194 +0.2798 a2 +0.0333 a3 +1.2104 a1^2 +0.5393 a1 a2 +0.1465 a1 a3 -1.7530 a2^2 +0.0569 a2 a3
