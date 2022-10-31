window.ChunkingBratConfig = {
    entity_types: [],
    relation_types: []
};

var tags = [
    {
        "subtags": ['NP'],
        "color": "#b4bbff"
    },
    {
        "subtags": ['PP'],
        "color": "#6ec1e2"
    },
    {
        "subtags": ['VP'],
        "color": "#adf6a2"
    },
    {
        "subtags": ['AP'],
        "color": "#f98fff"
    },
    {
        "subtags": ['M'],
        "color": "#fc6"
    },
    {
        "subtags": ['C', 'Cc', 'R', 'L', 'E', 'T', 'X'],
        "color": "#cc9"
    },
    {
        "subtags": ['CH'],
        "color": "#aaa"
    }
];

_.each(tags, function(tagCategory){
    var color = tagCategory["color"];
    _.each(tagCategory["subtags"], function(tag){
        var entity = {
            type: tag,
            labels: [tag],
            bgColor: color,
            borderColor: 'darken'
        };
        window.ChunkingBratConfig["entity_types"].push(entity);
    });
});

