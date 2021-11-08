# Forests As A Service
Gradient boosted trees.

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
