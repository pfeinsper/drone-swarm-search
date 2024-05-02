---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Drone Swarm Search Environment"
  # text: "A environment to train reinforcement learning agents"
  tagline: An environment to train reinforcement learning agents for search and rescue operations in maritime scenarios.
  image: /pics/drone.png
  actions:
    - theme: brand
      text: Documentation
      link: /docs
    - theme: alt
      text: Our Story
      link: /OurStory/index
    - theme: alt
      text: QuickStart
      link: /quickStart


features:
  - title: Based on PettingZoo's interface
    icon: 🦁
    details: Compatible with leading reinforcement learning libraries
    link: https://pettingzoo.farama.org/
    linkText: Farama

  - title: FLOSS for Maritime SAR
    icon: 🌊
    details: Possibly the only free/libre and open-source environment for maritime search and rescue operations.
    link: https://www.flossmanuals.net/
    linkText: Learn More

  - title: INSPER's College
    icon: 🏢
    details: >
      INSPER is a Brazilian higher education institution focused on business, economics, engineering, and law.
    link: https://www.insper.edu.br/
    linkText: INSPER Site

  - title: Partner Company - EMBRAER
    icon: ✈️
    details: >
      EMBRAER is a Brazilian aerospace company that produces commercial, military, executive and agricultural aircraft and provides aeronautical services.
    link: https://embraer.com/br/pt
    linkText: EMBRAER Site

members:
  - avatar: 'https://github.com/JorasOliveira.png'
    name: 'jorás Oliveira'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/JorasOliveira'
    - icon: 'linkedin'
      link: https://www.linkedin.com/in/jorasoliveira/

  - avatar: 'https://www.github.com/Pedro2712.png'
    name: 'Pedro Henrique Britto Aragão Andrade'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/Pedro2712'
    - icon: 'linkedin'
      link: 'https://www.linkedin.com/in/pedro-henrique-britto-aragao-andrade/'

  - avatar: 'https://github.com/RicardoRibeiroRodrigues.png'
    name: 'Ricardo Ribeiro Rodrigues'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/RicardoRibeiroRodrigues'
    - icon: 'linkedin'
      link: 'https://www.linkedin.com/in/ricardo-ribeiro-rodrigues-983b94196/'

  - avatar: 'https://github.com/renatex333.png'
    name: 'Renato Laffranchi'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/renatex333'
    - icon: 'linkedin'
      link: 'https://www.linkedin.com/in/renato-laffranchi-falcao/'

  - avatar: 'https://github.com/enricofd.png'
    name: 'Enrico Damiani'
    title: 'Author'
    links:
    - icon: 'github' 
      link: 'https://github.com/enricofd'

  - avatar: 'https://github.com/Manuel-castanares.png'
    name: 'Manuel castanares'
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
  
  - avatar: 'https://github.com/fbarth.png'
    name: Fabrício Barth
    title: 'Author'
    links:
    - icon: 'github'
      link: https://github.com/fbarth
    - icon: 'linkedin'
      link: 'https://www.linkedin.com/in/fbarth/'

  - avatar: 'https://media.licdn.com/dms/image/C4E03AQGn6DuuYwhjBw/profile-displayphoto-shrink_200_200/0/1564709106422?e=1718236800&v=beta&t=3JrDTb5QTF4k5qFZbQc3lK9sgSJbalH7Y3QD_rthXBE'
    name: Jose Fernando Basso Brancalion
    title: 'Author'
    links:
    - icon: 'linkedin'
      link: https://www.linkedin.com/in/jose-fernando-basso-brancalion/

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

