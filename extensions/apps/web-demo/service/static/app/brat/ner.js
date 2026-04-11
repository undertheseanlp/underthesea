window.NERTagBratConfig = {
    entity_types: [],
    relation_types: []
};

var tags = [
    {
        "subtags": ['PER'],
        "color": "#f96c62"
    },
    {
        "subtags": ["LOC"],
        "color": "#17AFC1"
    },
    {
        "subtags": ["ORG"],
        "color": "#1792C1"
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
        window.NERTagBratConfig["entity_types"].push(entity);
    });
});

