# Oxford IIIT Pet Dataset Neural Networks

## Estructura del proyecto

```shell
.
├── src
│   ├── model.py
│   ├── train.py
│   └── sample.py
├── data/
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

### Entrenamiento

```shell
python src/train.py --help
```

### Prepación de datos

```shell
python src/data.py --help
```

### Predicción

```shell
python src/sample.py --help
```

### Benchmark

```shell
python src/benchmark.py --help
```

## Licencia

Este proyecto se distribuye bajo la licencia [MIT](https://opensource.org/licenses/MIT). Para más información, ver el archivo [LICENSE](LICENSE).
