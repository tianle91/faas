# Forests As A Service
![flow](flow.svg)

## Types of configurations
Input validation can happen at pipelines.

x steps
- numeric features - passthrough
- categorical features - encode
- ts - seasonality*4

y steps
- target is numeric
	- target is positive - logtransform
		- correlated with categorical - standardscaler
		- correlated with numeric - numericscaler
		- else - passthrough
(kiv) target is categorical - encode

w steps
- ts - historical decay (?)
	- multivariate - normalize(date)
