# Forests As A Service
Gradient boosted trees.


# Pipelines
Given some pre-defined pipeline components.
```
XTransformers = [...]
YTransformers = [...]
WTransformers = [...]
```

Training looks like:
```
data -XTransformers-> data+x
data -YTransformers-> data+y
data -WTransformers-> data+w
data+x -> X, data+y -> y, data+w -> w
X, y, w -training-> model
```

Prediction looks like:
```
data -XTransformers-> data+x
data+x -> X, X, model -> y
data, y -JoinableByRowID-> data+y
data+y -YTransformers(inverse)-> data+predictions
```

# Tools
`faas.encoder`
- categorical features -> OrdinalEncoder

`faas.scaler`
- categorical features strongly correlated with target -> StandardScaler

`faas.weight`
- recent trends are more important -> HistoricalDecay
- performance among groups are equally important -> Normalize
