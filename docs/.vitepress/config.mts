import { defineConfig } from 'vitepress'

export default defineConfig({
  base: '/drone-swarm-search/',
  themeConfig: {
    logo: {
      light: '/pics/embraerBlack.png',
      dark: '/pics/embraerWhite.png'
    },

    search: {
      provider: 'local',
    },

    nav: [
      { text: 'Home', link: '/' },
      { text: 'Documentation', link: '/docs' },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/pfeinsper/drone-swarm-search', ariaLabel: 'Github link' },
      
      {
        icon: {
          svg: '<svg role="img" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><title>PyPI</title><path d="pics/python.svg"/></svg>'
        },
        link: 'https://pypi.org/project/DSSE/',
        ariaLabel: 'Pypi link'
      },
    ],
  },
  
  locales: {
    root: {
        label: 'English',
        lang: 'en-US',
        title: "DSSE",
        description: "Drone Swarm Search Training Environment",
        themeConfig: {
          // https://vitepress.dev/reference/default-theme-config    
          sidebar: [
            {
              text: 'Documentation',
              items: [
                {
                    collapsed: true,
                    text: 'Training Environment',
                    items: [
                      { text: 'About', link: '/docs#about' },
                      { text: 'Quick Start', link: '/docs#quick-start' },
                      { text: 'General Info', link: '/docs#general-info' },
                      { text: 'Built in Functions', link: '/docs#built-in-functions' },
                      { text: 'Person Movement', link: '/docs#person-movement' },
                      { text: 'License', link: '/docs#license' },
                    ]
                },
    
                {
                  collapsed: true,
                  text: 'Algorithms',
                  items: [
                        { text: 'Item A da Seção A', link: '...' },
                        { text: 'Item B da Seção B', link: '...' },
                  ]
                }
              ]
            }
          ],
          
          footer: {
            message: 'Published under the MIT License.',
            copyright: 'Copyright © 2023'
          },
        },
      },
    pt: {
        label: 'Português',
        lang: 'pt-br',
        link: '/pt-br/',
      },
  },
})
