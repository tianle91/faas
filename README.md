# Forests As A Service
Gradient boosted trees.

![flow](flow.svg)

# TODO

## Weights
1. Are recent trends more important? If so, use `faas.weight.HistoricalDecay`
2. Do we care equally about performance among groups (or dates)? If so, use `faas.weight.Normalize`.
