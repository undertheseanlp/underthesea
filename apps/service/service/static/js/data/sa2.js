/**
 * Created by rain on 1/8/2017.
 */
window.data["sa2"] = {
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
        "entity_attribute_types": [
            {
                type: 'entity',
                /* brat supports multi-valued attributes, but in our case we will only
                 use a single value and add a glyph to the visualisation to indicate
                 that the entity carries that attribute */
                "values": {
                    "LAPTOP": {"glyph": "LAP"}
                }
            },
            {
                type: 'aspect',
                /* brat supports multi-valued attributes, but in our case we will only
                 use a single value and add a glyph to the visualisation to indicate
                 that the entity carries that attribute */
                "values": {
                    "GENERAL": {"glyph": ":GENERAL"},
                    "OPERATION_PERFORMANCE": {"glyph": ":OP"}
                }
            }
        ],
        "relation_types": []
    },
    "doc": {
        text: "The So called laptop Runs to Slow and I hate it!\n⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ",
        entities: [
            ['T1', 'Entity', [[14, 20]]],
            ['T2', 'Attribute', [[21, 25]]],
            ['T3', 'NEG', [[29, 33]]],
            ['T4', 'NEG', [[40, 44]]],
            ['T5', 'Entity', [[45, 47]]],
            ['T6', 'NEG', [[49, 50]]],
            ['T7', 'NEG', [[51, 52]]],
        ],
        attributes: [
            ['A1', 'entity', 'T6', "LAPTOP"],
            ['A2', 'aspect', 'T6', "OPERATION_PERFORMANCE"],
            ['A3', 'entity', 'T7', "LAPTOP"],
            ['A4', 'aspect', 'T7', "GENERAL"]
        ]
    }
};