/**
 * Created by rain on 1/8/2017.
 */
window.data["ner1"] = {
    "config": {
        entity_types: [
            {
                type: 'Date',
                labels: ['Date'],
                bgColor: '#46cab2',
                borderColor: 'darken'
            },
            {
                type: 'Time',
                labels: ['Date'],
                bgColor: '#46cab2',
                borderColor: 'darken'
            },
            {
                type: 'Location',
                labels: ['Location'],
                bgColor: '#3DB4F8',
                borderColor: 'darken'
            },
            {
                type: 'Person',
                labels: ['Person'],
                bgColor: '#EEC467',
                borderColor: 'darken'
            }
        ],
        "relation_types": []
    },
    "doc": {
        text: "At the W party Thursday night at Chateau Marmont, Cate Blanchett barely made it up in the elevator.",
        entities: [
            ['T1', 'Date', [[15, 23]]],
            ['T2', 'Time', [[24, 29]]],
            ['T3', 'Location', [[32, 48]]],
            ['T4', 'Person', [[49, 64]]]
        ]
    }
};