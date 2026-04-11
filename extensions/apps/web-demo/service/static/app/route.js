app.config(function ($stateProvider, $urlRouterProvider) {
    $urlRouterProvider.otherwise('/');
    $stateProvider
        .state({
            url: '/',
            name: 'syntax',
            controller: 'WordSentCtrl',
            templateUrl: "./static/app/syntax/syntax.html"
        })
        .state({
            url: '/ner',
            name: 'ner',
            controller: 'NERTagCtrl',
            templateUrl: "./static/app/named_entity_recognition/ner.html"
        })
        .state({
            url: '/classification',
            name: 'classification',
            controller: 'ClassificationCtrl',
            templateUrl: "./static/app/classification/classification.html"
        })
        .state({
            url: '/sentiment',
            name: 'sentiment',
            controller: 'SentimentCtrl',
            templateUrl: "./static/app/sentiment/sentiment.html"
        })
        .state({
            url: '/dictionary',
            name: 'dictionary',
            controller: 'DictionaryCtrl',
            templateUrl: "./static/app/dictionary/dictionary.html"
        })
        .state({
            url: '/ipa',
            name: 'ipa',
            controller: 'IPACtrl',
            templateUrl: "./static/app/ipa/ipa.html"
        });
});
