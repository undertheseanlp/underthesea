from languageflow.transformer.tagged import TaggedTransformer
from languageflow.transformer.tagged_feature import word2features


class CustomTransformer(TaggedTransformer):
    def extract_features(self, feature):
        n = feature.find("=")
        return [feature[:n], feature[n+1:]]

    def _convert_features_to_dict(self, features):
        return dict([self.extract_features(feature) for feature in features])

    def _convert_features_to_list(self, features):
        return ["{}={}".format(k, v) for k, v in features.items()]

    def _word2features(self, s, i, template):
        features = word2features(s, i, template)
        features = self._convert_features_to_dict(features)
        return features

    def sentence2features(self, s):
        output = [self._word2features(s, i, self.template) for i in
                  range(len(s))]
        return output
