import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "DSSE",
  description: "Drone Swarm Search Training Environment",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Documentation', link: '/docs' },
      { text: 'Pypi', link: 'https://pypi.org/project/DSSE/'},
      { text: 'github', link: 'https://github.com/pfeinsper/drone-swarm-search'},
    ],

    sidebar: [
      {
        text: 'Documentation',
        items: [
          { text: 'About', link: '/docs#about' },
          { text: 'Quick Start', link: '/docs#quick-start' },
          { text: 'General Info', link: '/docs#general-info' },
          { text: 'Built in Functions', link: '/docs#built-in-functions' },
          { text: 'Person Movement', link: '/docs#person-movement' },
          // { text: 'Single page Docs', link: '/docs' },
        ]
      }
    ],
      
    socialLinks: [
      { icon: 'github', link: 'https://github.com/pfeinsper/drone-swarm-search' }
    ]
  }
})
