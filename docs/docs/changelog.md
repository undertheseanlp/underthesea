# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [9.2.0] - 2026-01-31

### Added

- Add Agent class with custom tools support using OpenAI function calling ([GH-712](https://github.com/undertheseanlp/underthesea/issues/712))
- Add default tools: calculator, datetime, web_search, wikipedia, shell, python, file operations ([GH-712](https://github.com/undertheseanlp/underthesea/issues/712))

### Changed

- Upgrade underthesea_core to 2.0.0 with L-BFGS optimizer ([#899](https://github.com/undertheseanlp/underthesea/pull/899))
  - 10x faster feature lookup with flat data structure
  - 1.24x faster than python-crfsuite for word segmentation
  - L-BFGS with OWL-QN for L1 regularization

## [9.1.5] - 2026-01-29

### Added

- Add Agent API with OpenAI and Azure OpenAI support ([GH-745](https://github.com/undertheseanlp/underthesea/issues/745), [#890](https://github.com/undertheseanlp/underthesea/pull/890))
- Add ParserTrainer for dependency parsing ([GH-392](https://github.com/undertheseanlp/underthesea/issues/392), [#880](https://github.com/undertheseanlp/underthesea/pull/880))
- Add POS tagger training pipeline ([GH-423](https://github.com/undertheseanlp/underthesea/issues/423), [#883](https://github.com/undertheseanlp/underthesea/pull/883))

### Documentation

- Add Vietnamese News Dataset (UVN) documentation ([GH-885](https://github.com/undertheseanlp/underthesea/issues/885), [#888](https://github.com/undertheseanlp/underthesea/pull/888), [#889](https://github.com/undertheseanlp/underthesea/pull/889))
- Add UVB dataset documentation ([GH-720](https://github.com/undertheseanlp/underthesea/issues/720), [#887](https://github.com/undertheseanlp/underthesea/pull/887))
- Add UUD-v0.1 dataset documentation ([#886](https://github.com/undertheseanlp/underthesea/pull/886))
- Add UTS Dictionary dataset documentation ([GH-622](https://github.com/undertheseanlp/underthesea/issues/622), [#884](https://github.com/undertheseanlp/underthesea/pull/884))

## [9.1.4] - 2026-01-24

### Added

- Implement Logistic Regression library in Rust ([#878](https://github.com/undertheseanlp/underthesea/pull/878))
- Implement CRF library in Rust ([#876](https://github.com/undertheseanlp/underthesea/pull/876))

### Changed

- Remove NLTK dependency ([#879](https://github.com/undertheseanlp/underthesea/pull/879))

### Security

- Fix Dependabot security vulnerabilities ([#874](https://github.com/undertheseanlp/underthesea/pull/874), [#875](https://github.com/undertheseanlp/underthesea/pull/875))

## [9.1.3] - 2026-01-24

### Added

- Add dependency tree visualization ([#867](https://github.com/undertheseanlp/underthesea/pull/867))

### Changed

- Support PyTorch v2 for dependency parsing ([#871](https://github.com/undertheseanlp/underthesea/pull/871))
- Update CP_Vietnamese-VLC README with HuggingFace dataset ([#872](https://github.com/undertheseanlp/underthesea/pull/872))

### Fixed

- Fix ValueError when loading DependencyParser from non-existent path ([#873](https://github.com/undertheseanlp/underthesea/pull/873))
- Fix KeyError in Sentence.__getattr__ ([#870](https://github.com/undertheseanlp/underthesea/pull/870))
- Fix TTS UnicodeDecodeError on Windows ([#869](https://github.com/undertheseanlp/underthesea/pull/869))
- Fix underthesea[voice] installation ([#868](https://github.com/undertheseanlp/underthesea/pull/868))

## [9.1.2] - 2026-01-24

### Added

- Add `labels` property to `classify` and `sentiment` functions ([#865](https://github.com/undertheseanlp/underthesea/pull/865))

### Fixed

- Fix sklearn >= 1.5 compatibility for loaded models ([#866](https://github.com/undertheseanlp/underthesea/pull/866))

## [9.1.1] - 2026-01-24

### Fixed

- Fix VERSION file to match pyproject.toml

## [9.1.0] - 2026-01-24

### Added

- Vietnamese-English translation module with `translate()` function ([#856](https://github.com/undertheseanlp/underthesea/pull/856))
- English to Vietnamese translation example in README ([#858](https://github.com/undertheseanlp/underthesea/pull/858))

### Changed

- Support Python 3.14, deprecate Python 3.9 ([#862](https://github.com/undertheseanlp/underthesea/pull/862))
- Migrate from Flake8/Pylint to Ruff for linting ([#857](https://github.com/undertheseanlp/underthesea/pull/857))

### Fixed

- Fix missing sdist (tar.gz) on PyPI for underthesea_core ([#859](https://github.com/undertheseanlp/underthesea/pull/859))

## [8.3.0] - 2025-09-28

### Added

- Train text classification model for dataset VNTC2017_BANK ([#819](https://github.com/undertheseanlp/underthesea/pull/819))
- Add datasets UTS2017_Bank ([#822](https://github.com/undertheseanlp/underthesea/pull/822))
- Add bank model ([#824](https://github.com/undertheseanlp/underthesea/pull/824))
- Build wheels for macOS x86-64 ([#820](https://github.com/undertheseanlp/underthesea/pull/820))

### Removed

- Remove flake8 as runtime dependency ([#818](https://github.com/undertheseanlp/underthesea/pull/818))

## [8.2.0] - 2025-09-21

### Changed

- Update project structure, create extensions/lab folder ([#812](https://github.com/undertheseanlp/underthesea/pull/812))
- Create Sonar Core 1 - System Card ([#813](https://github.com/undertheseanlp/underthesea/pull/813))
- Update output format of model sonar_core_1 ([#815](https://github.com/undertheseanlp/underthesea/pull/815))

## [8.1.0] - 2025-09-21

### Fixed

- Fix missing .pkl files ([#809](https://github.com/undertheseanlp/underthesea/pull/809))

## [8.0.1] - 2025-09-21

### Fixed

- Fix missing .txt files ([#806](https://github.com/undertheseanlp/underthesea/pull/806))

### Changed

- Update publish distribution to PyPI workflow ([#805](https://github.com/undertheseanlp/underthesea/pull/805))

### Security

- Security updates for dependencies

## [8.0.0] - 2025-09-20

### Added

- Underthesea Languages v2 ([#748](https://github.com/undertheseanlp/underthesea/pull/748))
- Interactive Page for Most Frequently Used Vietnamese Words ([#756](https://github.com/undertheseanlp/underthesea/pull/756))
- Support Python 3.12, 3.13 ([#777](https://github.com/undertheseanlp/underthesea/pull/777))

### Changed

- Update PyO3 API usage ([#768](https://github.com/undertheseanlp/underthesea/pull/768))
- Update project structure ([#790](https://github.com/undertheseanlp/underthesea/pull/790))

### Fixed

- Fix wrong global var in sent_tokenize ([#764](https://github.com/undertheseanlp/underthesea/pull/764))
- Fix logo in Readme.rst ([#761](https://github.com/undertheseanlp/underthesea/pull/761))

## [6.8.4] - 2024-06-22

### Added

- Add lang_detect module ([#733](https://github.com/undertheseanlp/underthesea/pull/733))

### Changed

- Optimize imports ([#741](https://github.com/undertheseanlp/underthesea/pull/741))
- Remove issue-manager workflow ([#726](https://github.com/undertheseanlp/underthesea/pull/726))

## [6.8.0] - 2023-09-23

### Added

- Release Source Distribution for underthesea_core ([#708](https://github.com/undertheseanlp/underthesea/pull/708))
- Create docker image for underthesea ([#711](https://github.com/undertheseanlp/underthesea/pull/711))

### Changed

- Code refactoring ([#713](https://github.com/undertheseanlp/underthesea/pull/713))

### Fixed

- Fix permission errors on removing downloaded models ([#715](https://github.com/undertheseanlp/underthesea/pull/715))

## [6.7.0] - 2023-07-28

### Added

- Zero shot classification with OpenAI API ([#700](https://github.com/undertheseanlp/underthesea/pull/700))

## [6.6.0] - 2023-07-27

### Fixed

- Fix bug word_tokenize ([#697](https://github.com/undertheseanlp/underthesea/pull/697))

## [6.5.0] - 2023-07-14

### Fixed

- Fix text_normalizer token rules

## [6.4.0] - 2023-07-14

### Fixed

- Fix fixed_words regex

## [6.3.0] - 2023-06-28

### Added

- Support MacOS ARM

## [6.2.0] - 2023-03-04

### Added

- Add Text to Speech API ([#668](https://github.com/undertheseanlp/underthesea/pull/668))
- Provide training script for word segmentation, pos tagging, and NER ([#666](https://github.com/undertheseanlp/underthesea/pull/666))
- Create UTS_Dictionary v1.0 datasets ([#663](https://github.com/undertheseanlp/underthesea/pull/663))

## [6.1.4] - 2023-02-26

### Added

- Support underthesea_core with Python 3.11 ([#659](https://github.com/undertheseanlp/underthesea/pull/659))

## [6.1.2] - 2023-02-15

### Added

- Add option fixed_words to tokenize and word_tokenize API ([#649](https://github.com/undertheseanlp/underthesea/pull/649))

## [6.0.0] - 2023-01-01

### Changed

- Version bump for 2023

## [1.4.1] - 2022-12-17

### Added

- Create underthesea app
- Add viet2ipa module
- Training NER model with VLSP2016 dataset using BERT

### Removed

- Remove unidecode as a dependency

## [1.3.5] - 2022-10-31

### Added

- Add Text Normalization module
- Release underthesea_core version 0.0.5a2
- Support GLIBC_2.17

### Changed

- Update resources path

### Fixed

- Fix function word_tokenize

## [1.3.4] - 2022-01-08

### Added

- Demo chatbot with rasa
- Lite version of underthesea
- Add build for Windows

### Changed

- Increase word_tokenize speed 1.5 times

## [1.3.3] - 2021-09-02

### Changed

- Update torch and transformer dependency

## [1.3.2] - 2021-08-04

### Added

- Publish two ABSA open datasets
- Add pipeline folder

### Changed

- Migrate from travis-ci to github actions
- Update ParserTrainer

## [1.3.1] - 2021-01-11

### Added

- Add ClassifierTrainer
- Add 3 new datasets

### Changed

- Compatible with newer version of scikit-learn
- Retrain classification and sentiment models

## [1.3.0] - 2020-12-11

### Added

- Dependency Parsing

### Removed

- Remove languageflow dependency
- Remove tabulate dependency

## [1.0.0] - 2017-03-01

### Added

- First release on PyPI
- First release on ReadTheDocs

[Unreleased]: https://github.com/undertheseanlp/underthesea/compare/v9.1.5...HEAD
[9.1.5]: https://github.com/undertheseanlp/underthesea/compare/v9.1.4...v9.1.5
[9.1.4]: https://github.com/undertheseanlp/underthesea/compare/v9.1.3...v9.1.4
[9.1.3]: https://github.com/undertheseanlp/underthesea/compare/v9.1.2...v9.1.3
[9.1.2]: https://github.com/undertheseanlp/underthesea/compare/v9.1.1...v9.1.2
[9.1.1]: https://github.com/undertheseanlp/underthesea/compare/v9.1.0...v9.1.1
[9.1.0]: https://github.com/undertheseanlp/underthesea/compare/v8.3.0...v9.1.0
[8.3.0]: https://github.com/undertheseanlp/underthesea/compare/v8.2.0...v8.3.0
[8.2.0]: https://github.com/undertheseanlp/underthesea/compare/v8.1.0...v8.2.0
[8.1.0]: https://github.com/undertheseanlp/underthesea/compare/v8.0.1...v8.1.0
[8.0.1]: https://github.com/undertheseanlp/underthesea/compare/v8.0.0...v8.0.1
[8.0.0]: https://github.com/undertheseanlp/underthesea/compare/v6.8.4...v8.0.0
[6.8.4]: https://github.com/undertheseanlp/underthesea/compare/v6.8.0...v6.8.4
[6.8.0]: https://github.com/undertheseanlp/underthesea/compare/v6.7.0...v6.8.0
[6.7.0]: https://github.com/undertheseanlp/underthesea/compare/v6.6.0...v6.7.0
[6.6.0]: https://github.com/undertheseanlp/underthesea/compare/v6.5.0...v6.6.0
[6.5.0]: https://github.com/undertheseanlp/underthesea/compare/v6.4.0...v6.5.0
[6.4.0]: https://github.com/undertheseanlp/underthesea/compare/v6.3.0...v6.4.0
[6.3.0]: https://github.com/undertheseanlp/underthesea/compare/v6.2.0...v6.3.0
[6.2.0]: https://github.com/undertheseanlp/underthesea/compare/v6.1.4...v6.2.0
[6.1.4]: https://github.com/undertheseanlp/underthesea/compare/v6.1.2...v6.1.4
[6.1.2]: https://github.com/undertheseanlp/underthesea/compare/v6.0.0...v6.1.2
[6.0.0]: https://github.com/undertheseanlp/underthesea/compare/v1.4.1...v6.0.0
[1.4.1]: https://github.com/undertheseanlp/underthesea/compare/v1.3.5...v1.4.1
[1.3.5]: https://github.com/undertheseanlp/underthesea/compare/v1.3.4...v1.3.5
[1.3.4]: https://github.com/undertheseanlp/underthesea/compare/v1.3.3...v1.3.4
[1.3.3]: https://github.com/undertheseanlp/underthesea/compare/v1.3.2...v1.3.3
[1.3.2]: https://github.com/undertheseanlp/underthesea/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/undertheseanlp/underthesea/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/undertheseanlp/underthesea/compare/v1.0.0...v1.3.0
[1.0.0]: https://github.com/undertheseanlp/underthesea/releases/tag/v1.0.0
