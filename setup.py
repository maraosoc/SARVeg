from setuptools import setup, find_packages

setup(
    name="sar_vegetation",          # Nombre del paquete
    version="0.1",                  # Versión inicial
    packages=find_packages("src"),  # Encuentra automáticamente todos los subpaquetes en src/
    package_dir={"": "src"},        # Indica que el código fuente está en src/
    install_requires=[              # Librerías requeridas
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "lightgbm",
        "statsmodels",
        "joblib",
        "geopandas",
        "rasterio",
        "shapely"
    ],
)
