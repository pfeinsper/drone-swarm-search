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
      { text: 'Documentation', link: '/Documentation/docsSearch' },
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/pfeinsper/drone-swarm-search', ariaLabel: 'Github link' },
      
      {
        icon: {
          svg: '<svg width="800px" height="800px" viewBox="15 15 32 32" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M31.885 16c-8.124 0-7.617 3.523-7.617 3.523l.01 3.65h7.752v1.095H21.197S16 23.678 16 31.876c0 8.196 4.537 7.906 4.537 7.906h2.708v-3.804s-.146-4.537 4.465-4.537h7.688s4.32.07 4.32-4.175v-7.019S40.374 16 31.885 16zm-4.275 2.454c.771 0 1.395.624 1.395 1.395s-.624 1.395-1.395 1.395a1.393 1.393 0 0 1-1.395-1.395c0-.771.624-1.395 1.395-1.395z" fill="url(#a)"/><path d="M32.115 47.833c8.124 0 7.617-3.523 7.617-3.523l-.01-3.65H31.97v-1.095h10.832S48 40.155 48 31.958c0-8.197-4.537-7.906-4.537-7.906h-2.708v3.803s.146 4.537-4.465 4.537h-7.688s-4.32-.07-4.32 4.175v7.019s-.656 4.247 7.833 4.247zm4.275-2.454a1.393 1.393 0 0 1-1.395-1.395c0-.77.624-1.394 1.395-1.394s1.395.623 1.395 1.394c0 .772-.624 1.395-1.395 1.395z" fill="url(#b)"/><defs><linearGradient id="a" x1="19.075" y1="18.782" x2="34.898" y2="34.658" gradientUnits="userSpaceOnUse"><stop stop-color="#387EB8"/><stop offset="1" stop-color="#366994"/></linearGradient><linearGradient id="b" x1="28.809" y1="28.882" x2="45.803" y2="45.163" gradientUnits="userSpaceOnUse"><stop stop-color="#FFE052"/><stop offset="1" stop-color="#FFC331"/></linearGradient></defs></svg>'
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
          sidebar: {
                "/Documentation/": [{
                    collapsed: false,
                    text: 'Search Environment',
                    items: [
                      { text: 'About', link: '/Documentation/docsSearch#about' },
                      { text: 'Quick Start', link: '/Documentation/docsSearch#quick-start' },
                      { text: 'General Info', link: '/Documentation/docsSearch#general-info' },
                      { text: 'Built in Functions', link: '/Documentation/docsSearch#built-in-functions' },
                      { text: 'Person Movement', link: '/Documentation/docsSearch#person-movement' },
                      { text: 'Observation', link: '/Documentation/docsSearch#observation' },
                      { text: 'Reward', link: '/Documentation/docsSearch#reward' },
                      { text: 'Termination & Truncation', link: '/Documentation/docsSearch#termination-truncation' },
                      { text: 'Info', link: '/Documentation/docsSearch#info' },
                      { text: 'How to Cite This Work', link: '/Documentation/docsSearch#how-to-cite-this-work' },
                      { text: 'License', link: '/Documentation/docsSearch#license' },
                    ]
                },
                
                {
                  collapsed: false,
                  text: 'Coverage Environment',
                  items: [
                    { text: 'About', link: '/Documentation/docsCoverage#about' },
                    { text: 'Quick Start', link: '/Documentation/docsCoverage#quick-start' },
                    { text: 'General Info', link: '/Documentation/docsCoverage#general-info' },
                    { text: 'Built in Functions', link: '/Documentation/docsCoverage#built-in-functions' },
                    { text: 'Probability Matrix', link: '/Documentation/docsCoverage#probability-matrix' },
                    { text: 'Reward', link: '/Documentation/docsCoverage#reward' },
                    { text: 'Termination & Truncation', link: '/Documentation/docsCoverage#termination-truncation' },
                    { text: 'Info', link: '/Documentation/docsCoverage#info' },
                    { text: 'License', link: '/Documentation/docsCoverage#license' },
                  ]
                },
                // {
                //   collapsed: false,
                //   text: 'Algorithms',
                //   items: [
                //     { text: 'Documentation Under Construction', link: '/Documentation/docsAlgorithms#build' },
                //   ]
                // }
              ],
              "/QuickStart/": [{
                  collapsed: false,
                  text: 'Quick Start',
                  items: [
                    { text: 'Quick Start Guide', link: '/QuickStart/quickStart#quick-start-guide' },
                    { text: 'Search Environment', link: '/QuickStart/quickStart#search-environment' },
                    { text: 'Coverage Environment', link: '/QuickStart/quickStart#coverage-environment' },
                  ]
              },],
              "/OurStory/": [{
                    collapsed: false,
                    text: 'Our Story',
                    items: [
                      { text: 'Our Story Coming Soon!', link: '/OurStory/ourStory#our-story-coming-soon!' },
                      { text: 'Stay Updated', link: '/OurStory/ourStory#stay-updated' },
                    ]
                },],
          },
          
          footer: {
            message: 'Published under the MIT License.',
            copyright: 'Copyright © 2023'
          },
        },
      },
    pt: {
        label: 'Português (Em breve)',
        lang: 'pt-br',
        link: '/pt/',
      },
  },
})
