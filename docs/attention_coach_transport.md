# Attention Coach Transport

## Goal

Test whether a frozen public-data model transfers prospectively to limited Attention Coach data without refitting the core representation.

## Transport Contract

- Freeze features, preprocessing and model selection before inspecting target outcomes.
- Use only compatible behavioural signals available in Attention Coach.
- Report missingness and feature incompatibilities explicitly.
- Score calibration and predictive utility against simple baselines.

## Claims Boundary

A successful transport test supports portability of the behavioural model. It does not establish clinical validity, neural mechanism, or durable intervention benefit.

