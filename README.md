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
data+x -> X
X -model-> y
data, y -JoinableByRowID-> data+y
data+y -YTransformers(inverse)-> data+predictions
```

# Steps
1. Identify target columns. If target is categorical, use `faas.encoder.OrdinalEncoder`.
2. Identify which columns to use as predictive features? For any categorical columns, use 
`faas.encoder.OrdinalEncoder`.

## Numeric Target
1. Any features strongly correlated with target? For categorical features, use
`faas.scaler.StandardScaler`. For numeric features, use `faas.scaler.NumericScaler`.

## Binary Target
1. Are the classes imbalanced? TBD.

## Weights
1. Are recent trends more important? If so, use `faas.weight.HistoricalDecay`
2. Do we care equally about performance among groups (or dates)? If so, use `faas.weight.Normalize`.
