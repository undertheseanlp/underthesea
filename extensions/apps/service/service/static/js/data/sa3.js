/**
 * Created by rain on 1/8/2017.
 */
window.data["sa3"] = {
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
                    "AMBIENCE": {"glyph": "AMB"},
                    "RESTAURANT": {"glyph": "RES"},
                }
            },
            {
                type: 'aspect',
                /* brat supports multi-valued attributes, but in our case we will only
                 use a single value and add a glyph to the visualisation to indicate
                 that the entity carries that attribute */
                "values": {
                    "GENERAL": {"glyph": ":GEN"},
                    "OPERATION_PERFORMANCE": {"glyph": ":OP"},
                    "PRICES": {"glyph": ":PRI"},
                }
            }
        ],
        "relation_types": []
    },
    "doc": {
        text: "This little place has a cute interior decor and affordable city prices. The pad seew chicken was delicious, however the pad thai was far too oily. I would just ask for no oil next time.\n⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ⍟ ",
        entities: [
            ['OTE1', 'POS', [[186, 187]]],
            ['OTE2', 'POS', [[188, 189]]],
            ['OTE3', 'POS', [[190, 191]]],
            ['E1', 'Entity', [[12, 17]]],
            ['O1', 'POS', [[24, 28]]],
            ['A1', 'Attribute', [[29, 42]]],
            ['O2', 'POS', [[48, 58]]],
            ['A2', 'Attribute', [[64, 70]]]
        ],
        attributes: [
            ['A1', 'entity', 'OTE1', "AMBIENCE"],
            ['A1', 'aspect', 'OTE1', "GENERAL"],
            ['A2', 'entity', 'OTE2', "RESTAURANT"],
            ['A2', 'aspect', 'OTE2', "PRICES"],
            ['A2', 'entity', 'OTE3', "RESTAURANT"],
            ['A2', 'aspect', 'OTE3', "GENERAL"],
        ]
    }
};