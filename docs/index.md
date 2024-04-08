---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Drone Swarm Search Environment"
  # text: "A environment to train reinforcement learning agents"
  tagline: A environment to train reinforcement learning agents for search and rescue operations in maritime scenarios.
  actions:
    - theme: brand
      text: Documentation
      link: /docs
    - theme: alt
      text: QuickStart
      link: /quickStart
    - theme: alt
      text: pypi
      link: https://pypi.org/project/DSSE/

features:
  # - title: Based on petting zoo and OpenAI Gymmnasium
  #   details: Compatible with know RL libraries
  # - title: FLOSS
  #   details: maybe the only FLOSS env for maritime SAR operations?
  - title: Fabrício Barth
    details: PhD, Computer Engineering
    link: https://github.com/fbarth
    linkText: GitHub
  - title: Jose Fernando Basso Brancalion
    details: Product Development Engineer, Embraer
    link: https://www.linkedin.com/in/jose-fernando-basso-brancalion/
    linkText: Linkein
  # - title: Feature C
  #   details: Lorem ipsum dolor sit amet, consectetur adipiscing elit

members:
  - avatar: 'https://github.com/JorasOliveira.png'
    name: 'jorás Oliveira'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/JorasOliveira'

  - avatar: 'https://www.github.com/Pedro2712.png'
    name: 'Pedro Andrade'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/Pedro2712'

  - avatar: 'https://github.com/RicardoRibeiroRodrigues.png'
    name: 'Ricardo Rodrigues'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/RicardoRibeiroRodrigues'

  - avatar: 'https://github.com/renatex333.png'
    name: 'Renato Laffranchi.'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/renatex333'

  - avatar: 'https://github.com/enricofd.png'
    name: 'Enrico Damiani'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/enricofd'

  - avatar: 'https://github.com/Manuel-castanares.png'
    name: 'Manuel-castanares'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/Manuel-castanares'

  - avatar: 'https://github.com/lfcarrete.png'
    name: 'Luis Filipe Carrete'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/lfcarrete'

  - avatar: 'https://github.com/leonardodma.png'
    name: 'Leonardo Malta'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/leonardodma'

---

<script setup>
  import {
  VPTeamPage,
  VPTeamPageTitle,
  VPTeamMembers
} from 'vitepress/theme'
</script>

<VPTeamPage class="VPHomeDocTeamPage">
  <VPTeamMembers size="small" :members="$frontmatter.members" />
</VPTeamPage>

