# Forests As A Service
Gradient boosted trees.

```
data -|-> X (encoded covariates) -|-> X, y, w
      |-> y (normalized target) --|
      |-> weights ----------------|

X, y, w -> model -> serialized
```
