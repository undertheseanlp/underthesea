# Changelog

All notable changes to Underthesea are documented here.

## 9.0.0 (2025-01-XX)

### New Features

- Vietnamese-English translation module ([#856](https://github.com/undertheseanlp/underthesea/pull/856))
  - `translate()` function for VI→EN and EN→VI translation
  - Requires `[deep]` installation

### Improvements

- Migrated from Flake8/Pylint to Ruff for linting ([#857](https://github.com/undertheseanlp/underthesea/pull/857))

---

## 8.3.0 (2025-09-28)

- Remove flake8 as runtime dependency by @BLKSerene in [#818](https://github.com/undertheseanlp/underthesea/pull/818)
- Train text classification model for dataset VNTC2017_BANK by @rain1024 in [#819](https://github.com/undertheseanlp/underthesea/pull/819)
- Build wheels for macOS x86-64 by @BLKSerene in [#820](https://github.com/undertheseanlp/underthesea/pull/820)
- Add datasets UTS2017_Bank by @rain1024 in [#822](https://github.com/undertheseanlp/underthesea/pull/822)
- Add bank model by @rain1024 in [#824](https://github.com/undertheseanlp/underthesea/pull/824)

## 8.2.0 (2025-09-21)

- Update project structure, create extensions/lab folder by @rain1024 in [#812](https://github.com/undertheseanlp/underthesea/pull/812)
- Create Sonar Core 1 - System Card by @rain1024 in [#813](https://github.com/undertheseanlp/underthesea/pull/813)
- Update output format of model sonar_core_1 by @rain1024 in [#815](https://github.com/undertheseanlp/underthesea/pull/815)

## 8.1.0 (2025-09-21)

- Fix missing .pkl files by @rain1024 in [#809](https://github.com/undertheseanlp/underthesea/pull/809)

## 8.0.1 (2025-09-21)

- Security updates for dependencies
- Update publish distribution to Pypi workflow by @rain1024 in [#805](https://github.com/undertheseanlp/underthesea/pull/805)
- Fix missing .txt files by @rain1024 in [#806](https://github.com/undertheseanlp/underthesea/pull/806)

## 8.0.0 (2025-09-20)

### New Features

- Underthesea Languages v2 by @rain1024 in [#748](https://github.com/undertheseanlp/underthesea/pull/748)
- Interactive Page for Most Frequently Used Vietnamese Words by @rain1024 in [#756](https://github.com/undertheseanlp/underthesea/pull/756)

### Improvements

- Support Python 3.12, 3.13 by @rain1024 in [#777](https://github.com/undertheseanlp/underthesea/pull/777)
- Update PyO3 API usage by @trunghieu0207 in [#768](https://github.com/undertheseanlp/underthesea/pull/768)
- Update project structure by @rain1024 in [#790](https://github.com/undertheseanlp/underthesea/pull/790)

### Bug Fixes

- Fix wrong global var in sent_tokenize by @Darejkal in [#764](https://github.com/undertheseanlp/underthesea/pull/764)
- Minor fix (Logo in Readme.rst) by @ichxorya in [#761](https://github.com/undertheseanlp/underthesea/pull/761)

---

## 6.8.4 (2024-06-22)

- Optimize imports by @rain1024 in [#741](https://github.com/undertheseanlp/underthesea/pull/741)
- Remove issue-manager workflow by @rain1024 in [#726](https://github.com/undertheseanlp/underthesea/pull/726)
- Add lang_detect module by @rain1024 in [#733](https://github.com/undertheseanlp/underthesea/pull/733)

## 6.8.0 (2023-09-23)

- Release Source Distribution for underthesea_core by @rain1024 in [#708](https://github.com/undertheseanlp/underthesea/pull/708)
- Create docker image for underthesea by @rain1024 in [#711](https://github.com/undertheseanlp/underthesea/pull/711)
- Code refactoring by @tosemml in [#713](https://github.com/undertheseanlp/underthesea/pull/713)
- Fix permission errors on removing downloaded models by @BLKSerene in [#715](https://github.com/undertheseanlp/underthesea/pull/715)

## 6.7.0 (2023-07-28)

- Zero shot classification with OpenAI API by @rain1024 in [#700](https://github.com/undertheseanlp/underthesea/pull/700)

## 6.6.0 (2023-07-27)

- Fix bug word_tokenize by @rain1024 in [#697](https://github.com/undertheseanlp/underthesea/pull/697)

## 6.5.0 (2023-07-14)

- Fix text_normalizer token rules

## 6.4.0 (2023-07-14)

- Fix fixed_words regex

## 6.3.0 (2023-06-28)

- Support MacOS ARM

## 6.2.0 (2023-03-04)

- Add Text to Speech API by @rain1024 in [#668](https://github.com/undertheseanlp/underthesea/pull/668)
- Provide training script for word segmentation, pos tagging, and NER by @rain1024 in [#666](https://github.com/undertheseanlp/underthesea/pull/666)
- Create UTS_Dictionary v1.0 datasets by @rain1024 in [#663](https://github.com/undertheseanlp/underthesea/pull/663)

## 6.1.4 (2023-02-26)

- Support underthesea_core with Python 3.11 by @rain1024 in [#659](https://github.com/undertheseanlp/underthesea/pull/659)

## 6.1.2 (2023-02-15)

- Add option fixed_words to tokenize and word_tokenize API by @rain1024 in [#649](https://github.com/undertheseanlp/underthesea/pull/649)

## 6.0.0 (2023-01-01)

- Happy New Year 2023! Version bump!

---

## 1.4.1 (2022-12-17)

- Create underthesea app
- Add viet2ipa module
- Training NER model with VLSP2016 dataset using BERT
- Remove unidecode as a dependency

## 1.3.5 (2022-10-31)

- Add Text Normalization module
- Release underthesea_core version 0.0.5a2
- Support GLIBC_2.17
- Update resources path
- Fix function word_tokenize

## 1.3.4 (2022-01-08)

- Demo chatbot with rasa
- Lite version of underthesea
- Increase word_tokenize speed 1.5 times
- Add build for Windows

## 1.3.3 (2021-09-02)

- Update torch and transformer dependency

## 1.3.2 (2021-08-04)

- Publish two ABSA open datasets
- Migrate from travis-ci to github actions
- Update ParserTrainer
- Add pipeline folder

## 1.3.1 (2021-01-11)

- Compatible with newer version of scikit-learn
- Retrain classification and sentiment models
- Add ClassifierTrainer
- Add 3 new datasets

## 1.3.0 (2020-12-11)

- Remove languageflow dependency
- Remove tabulate dependency
- Dependency Parsing

---

## 1.0.0 (2017-03-01)

- First release on PyPI
- First release on ReadTheDocs
