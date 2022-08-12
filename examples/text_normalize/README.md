# Text Normalization Module

## Usage

Install requirements

```
$ pip install -r requirements
```

Evaluation

```
$ python evaluation.py
$ python evaluation.py model=vtt
$ python evaluation.py model=nvh
$ python evaluation.py model=vtm
```

Make binary map files

```
python build.py
```

## Technical Report

* [Underthesea 1.3.5 - Text Normalization - WIP](reports/Underthesea_1_3_5___Text_Normalization__WIP.pdf)

## Appendix

Build VietnameseTextNormalizer

```
cp VietnameseTextNormalizer/ReleasePython3 underthesea/examples/text_normalize/tools/module
```
