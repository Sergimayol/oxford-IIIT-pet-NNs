# Oxford IIIT Pet Dataset Neural Networks

## Estructura del proyecto

```shell
.
├── src/
│   ├── model.py
│   ├── train.py
│   ├── data.py
│   ├── architectures.py
│   └── bench.py
├── data/
├── docs/
├── notebooks/
├── LICENSE
└── README.md
```

## Tareas (TODOs)

-   [x] Clasificación perro/gato
-   [x] Clasificación de razas
-   [x] Detección de cabezas
-   [ ] Segmentación de animales

## Requisitos

-   [Python](https://www.python.org/) 3.10 o superior
-   [Requirements.txt](requirements.txt)

```shell
pip install -r requirements.txt
```

## Uso

### Prepación de datos

```shell
python src/data.py -h
```

### Entrenamiento

```shell
python src/train.py -h
```

### Benchmark

```shell
python src/bench.py -h
```

## Arquitectura

```shell
python src/architectures.py
```

## Licencia

Este proyecto se distribuye bajo la licencia [MIT](https://opensource.org/licenses/MIT). Para más información, ver el archivo [LICENSE](LICENSE).
