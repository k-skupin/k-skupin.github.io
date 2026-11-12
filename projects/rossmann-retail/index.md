---
title: Rossmann Retail Sales Forecasting
description: Store-level sales forecasts with chronological validation and a command-line batch workflow.
---

How accurately can daily sales be forecast for individual stores using
information available before the forecast date?

This independent portfolio project uses data from the
[Rossmann Store Sales competition on Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales).

I joined historical sales and store data in SQLite, then compared a
store-weekday baseline, Ridge regression, and a random forest with target
encoding. The analysis also tested whether a pre-Christmas calendar feature
could improve winter forecasts.

## Model evaluation

I evaluated the models across three 42-day windows using the same 934 stores
with complete data coverage. The evaluation metric was mean absolute error
(MAE) per open store-day.

## Batch forecasting

The command-line script loads the saved Ridge model, reads a store schedule
from CSV, and writes a daily sales forecast for each row. Closed store-days
receive a forecast of zero.

[View the code, notebook, and setup instructions on GitHub](https://github.com/k-skupin/Rossmann_retail)

[← Back to Data Science Portfolio](../../)