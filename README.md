# EM-DGCN: Identification of Autism Spectrum Disorder based on Edge Dropping and Multi-Atlas Features fusion

## Requirements

torch == 1.8.1+cu102

torch-cluster == 1.5.9

torch-geometric == 1.7.0

torch-scatter == 2.0.7

torch-sparse == 0.6.10

sklearn

nilearn

## Download ABIDE I dataset

```
python download_ABIDE.py
```

## Run the diagnosis framework
Run the construct-graph:

```
python construct_graph.py
```

Run the framework:

```
python main.py
```



