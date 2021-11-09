# Forests As A Service
Gradient boosted trees.


# Pipelines
Training
```
data -GetX-> X
     -GetY-> y
     -GetW-> w

X, y, w -training-> model
```

Prediction
```
data -GetX-> X
X, model -> y
data, y -GetY-> data[y]
```

# Tools
`faas.encoder`
- categorical features -> OrdinalEncoderSingle

`faas.scaler`
- categorical features strongly correlated with target -> StandardScaler

`faas.weight`
- recent trends are more important -> HistoricalDecay
- performance among groups are equally important -> Normalize
