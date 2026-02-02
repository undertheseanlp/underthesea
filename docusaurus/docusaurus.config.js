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
            label: 'Docs',
          },
          {
            to: '/docs/technical-reports/sent-tokenize',
            label: 'Technical Reports',
            position: 'left',
          },
          {
            to: '/docs/api',
            label: 'API Reference',
            position: 'left',
          },
          {
            to: '/docs/datasets/uts-vlc',
            label: 'Datasets',
            position: 'left',
          },
          {
            to: '/docs/changelog',
            label: 'Changelog',
            position: 'left',
          },
          {
            to: '/blog',
            label: 'Blog',
            position: 'left',
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
            title: 'Docs',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/intro',
              },
              {
                label: 'API Reference',
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
