import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "DSSE",
  description: "Drone Swarm Search Training Environment",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Documentation', link: '/docs' }
    ],

    sidebar: [
      {
        text: 'Documentation',
        items: [
          { text: 'Documentation', link: '/docs' },
        ]
      }
    ],
      
    socialLinks: [
      { icon: 'gitgub', link: 'https://github.com/pfeinsper/drone-swarm-search' }
    ]
  }
})
