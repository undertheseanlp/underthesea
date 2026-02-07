/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docsSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Developer',
      items: [
        'developer/contributing',
        'developer/architecture',
        'developer/releasing',
      ],
    },
  ],
  technicalReportsSidebar: [
    {
      type: 'category',
      label: 'Technical Reports',
      collapsible: false,
      items: [
        {
          type: 'category',
          label: 'NLP',
          items: [
            'technical-reports/text-normalize',
            'technical-reports/sent-tokenize',
            'technical-reports/nlp/tagging',
            'technical-reports/ner',
            'technical-reports/nlp/text-classification',
            'technical-reports/translate',
            'technical-reports/language-identification',
          ],
        },
        {
          type: 'category',
          label: 'Audio',
          items: [
            'technical-reports/audio/text-to-speech',
          ],
        },
        {
          type: 'category',
          label: 'Agents',
          items: [
            'technical-reports/agents/index',
            'technical-reports/agents/tools',
            'technical-reports/agents/comparison',
          ],
        },
      ],
    },
  ],
  apiReferenceSidebar: [
    {
      type: 'category',
      label: 'API Reference',
      collapsible: false,
      items: [
        'api/index',
        'api/sent-tokenize',
        'api/text-normalize',
        'api/word-tokenize',
        'api/pos-tag',
        'api/chunk',
        'api/dependency-parse',
        'api/ner',
        'api/classify',
        'api/sentiment',
        'api/translate',
        'api/lang-detect',
        'api/tts',
        'api/agent',
      ],
    },
    {
      type: 'category',
      label: 'underthesea_core',
      items: [
        'api/core/index',
        'api/core/crf',
        'api/core/lr',
        'api/core/text-classifier',
        'api/core/tfidf',
        'api/core/text-preprocessor',
      ],
    },
  ],
  datasetsSidebar: [
    {
      type: 'category',
      label: 'Datasets',
      collapsible: false,
      items: [
        'datasets/uts-vlc',
        'datasets/uud-v0.1',
        'datasets/uvb',
        'datasets/uvn',
        'datasets/uvw',
      ],
    },
  ],
  changelogSidebar: [
    'changelog',
  ],
};

export default sidebars;
