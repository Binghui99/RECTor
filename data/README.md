# Data layout

Raw traces are intentionally excluded from version control. Place them as follows:

```text
data/raw/<split>/
├── inflow/
│   └── <circuit>_<site>
└── outflow/
    └── <circuit>_<site>
```

Each trace is tab-separated: `timestamp<TAB>packet_size`. Matching ingress and egress
files must have the same name. Generated windows belong in `data/processed/<split>/`.

