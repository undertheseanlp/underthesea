from .tagged import TaggedTransformer
from .tagged_feature import word2features


class CustomTransformer(TaggedTransformer):
    def _convert_features_to_dict(self, features):
        return dict([k.split("=") for k in features])

    def _convert_features_to_list(self, features):
        return [u"{}={}".format(k, v) for k, v in features.items()]
        pass

    def _word2features(self, s, i, template):
        features = word2features(s, i, template)
        features = self._convert_features_to_dict(features)
        for i in range(-2, 3):
            t = "T[{}].is_in_dict".format(i)
            t2 = "T[{}]".format(i)
            t3 = "T[{}].lower".format(i)
            if features[t] == 'True':
                features[t2] = "-"
                features[t3] = "-"
        for i in range(-2, 2):
            t = "T[{},{}].is_in_dict".format(i, i + 1)
            t2 = "T[{},{}]".format(i, i + 1)
            if features[t] == 'True':
                features[t2] = "-"
        features = self._convert_features_to_list(features)

        return features

    def sentence2features(self, s):
        output = [self._word2features(s, i, self.template) for i in
                  range(len(s))]
        return output
