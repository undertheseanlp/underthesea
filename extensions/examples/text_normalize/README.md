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

# ViWiktionary

Download viwiktionary dataset

```
wget https://dumps.wikimedia.org/viwiktionary/20220801/viwiktionary-20220801-pages-articles-multistream.xml.bz2
bzip2 -d viwiktionary-20220801-pages-articles-multistream.xml.bz2

wget https://dumps.wikimedia.org/viwiktionary/20220801/viwiktionary-20220801-pages-articles-multistream-index.txt.bz2
bzip2 -d viwiktionary-20220801-pages-articles-multistream-index.txt.bz2
```

https://vi.wiktionary.org/wiki/B%E1%BA%A3n_m%E1%BA%ABu:vie-pron

https://vi.wiktionary.org/wiki/B%E1%BA%A3n_m%E1%BA%ABu:VieIPA