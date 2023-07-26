# Technical Report 

Vietnamese Word Segmentation (VWS) plays a pivotal role in various Natural Language Processing (NLP) tasks for the Vietnamese language. Segmentation can be thought of as the process of splitting a sequence of characters into meaningful chunks or "words." This is particularly challenging for Vietnamese due to its nature of combining multiple lexical units into single written forms, which can be confusing without proper context.

Over the past years, numerous models have been proposed to tackle this issue, with Conditional Random Fields (CRF) being one of the prominent ones due to its ability to consider the context for making segmentation decisions. This report presents our experiment leveraging the CRF model for the VWS task on the UTS_WTK dataset.

## Results

The table below captures the results of the Vietnamese Word Segmentation task using the Conditional Random Fields (CRF) model:

| Dataset         | Model      | F1 Score |
|:----------------|:-----------|---------:|
| UTS_WTK (1.0.0) | CRF        | 0.977    |