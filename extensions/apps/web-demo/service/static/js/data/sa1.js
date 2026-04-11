/**
 * Created by rain on 1/8/2017.
 */
window.data["sa1"] = {
    "config": {
        entity_types: [
            {
                type: 'Entity',
                labels: ['Entity', 'Ent'],
                bgColor: '#00B0FF',
                borderColor: 'darken'
            },
            {
                type: 'Attribute',
                labels: ['Attribute', 'Att'],
                bgColor: '#FFC400',
                borderColor: 'darken'
            },
            {
                type: 'NEG',
                labels: ['NEG'],
                bgColor: '#FF1744',
                borderColor: 'darken'
            },
            {
                type: 'POS',
                labels: ['POS'],
                bgColor: '#00E676',
                borderColor: 'darken'
            }
        ],
        "relation_types": []
    },
    "doc": {
        text: "The So called laptop Runs to Slow and I hate it!",
        entities: [
            ['T1', 'Entity', [[14, 20]]],
            ['T2', 'Attribute', [[21, 25]]],
            ['T3', 'NEG', [[29, 33]]],
            ['T4', 'NEG', [[40, 44]]],
            ['T5', 'Entity', [[45, 47]]]
        ]
    }
};