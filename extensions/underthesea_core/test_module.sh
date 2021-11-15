poetry run maturin build --release
pip uninstall -y underthesea-core
pip install target/wheels/underthesea_core-0.0.4_alpha.5-cp36-cp36m-manylinux2010_x86_64.whl
python lab_underthesea_core.py