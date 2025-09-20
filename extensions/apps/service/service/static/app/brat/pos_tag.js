window.POSTagBratConfig = {
    entity_types: [],
    relation_types: []
};

var tags = [
    {
        "subtags": ['N', 'Np', 'Nc', 'Nb', 'Nu', 'Ny'],
        "color": "#b4bbff"
    },
    {
        "subtags": ['P'],
        "color": "#6ec1e2"
    },
    {
        "subtags": ['V'],
        "color": "#adf6a2"
    },
    {
        "subtags": ['A', 'Ab'],
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
        window.POSTagBratConfig["entity_types"].push(entity);
    });
});

