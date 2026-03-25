from setuptools import find_packages, setup


setup(
    name="cm2ml-encodings-eval",
    version="0.1.0",
    description="Self-supervised node/edge classification on typed NetworkX graphs with text encoders and GNNs",
    packages=find_packages(include=["cm2ml_encodings_eval", "cm2ml_encodings_eval.*"]),
    install_requires=[
        "networkx>=3.2",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "torch>=2.1",
        "torch-geometric>=2.5",
        "transformers>=4.39",
        "PyYAML>=6.0",
    ],
    extras_require={"word2vec": ["gensim>=4.3"]},
    python_requires=">=3.10",
)
