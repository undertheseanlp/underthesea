// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Underthesea',
  tagline: 'Open-source Vietnamese Natural Language Processing Toolkit',
  favicon: 'img/logo.png',

  url: 'https://undertheseanlp.github.io',
  baseUrl: process.env.BASE_URL || '/',

  organizationName: 'undertheseanlp',
  projectName: 'underthesea',
  trailingSlash: false,

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/undertheseanlp/underthesea/tree/main/docusaurus/',
          lastVersion: 'current',
          versions: {
            current: {
              label: '9.2.11',
            },
          },
        },
        blog: {
          showReadingTime: true,
          editUrl: 'https://github.com/undertheseanlp/underthesea/tree/main/docusaurus/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/logo.png',
      navbar: {
        title: 'Underthesea',
        logo: {
          alt: 'Underthesea Logo',
          src: 'img/logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Install',
          },
          {
            type: 'docSidebar',
            sidebarId: 'technicalReportsSidebar',
            label: 'Models',
            position: 'left',
          },
          {
            type: 'docSidebar',
            sidebarId: 'datasetsSidebar',
            label: 'Datasets',
            position: 'left',
          },
          {
            type: 'docSidebar',
            sidebarId: 'apiReferenceSidebar',
            label: 'API',
            position: 'left',
          },
          {
            type: 'docSidebar',
            sidebarId: 'changelogSidebar',
            label: 'Changelog',
            position: 'left',
          },
          {
            to: '/blog',
            label: 'Blog',
            position: 'left',
          },
          {
            type: 'docsVersionDropdown',
            position: 'right',
          },
          {
            href: 'https://github.com/undertheseanlp/underthesea',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Install',
            items: [
              {
                label: 'Getting Started',
                to: '/docs',
              },
              {
                label: 'API',
                to: '/docs/api',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Facebook',
                href: 'https://www.facebook.com/undertheseanlp/',
              },
              {
                label: 'YouTube',
                href: 'https://www.youtube.com/channel/UC9Jv1Qg49uprg6SjkyAqs9A',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'Blog',
                to: '/blog',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/undertheseanlp/underthesea',
              },
              {
                label: 'Google Colab',
                href: 'https://colab.research.google.com/drive/1gD8dSMSE_uNacW4qJ-NSnvRT85xo9ZY2',
              },
            ],
          },
        ],
        copyright: `Copyright © 2018 - ${new Date().getFullYear()} Underthesea. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash'],
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
    }),
};

export default config;
