# Paper para o Journal of Open Source Software

## Requisitos de Software

- [x] Software é Open Source segundo a [deifinção formal](https://opensource.org/osd).
- [x] Software é hospedado de forma que não requer aprovação ou pagamentos.
- [x] Software possui aplicação óbvia em pesquisa. (No nosso caso, "supports the functioning of research instruments or the execution of research experiments".)
- [x] O *submitter* deve ser contribuidor majoritário. (Não menos de 3 meses de trabalho.)
- [] O paper não deve focar nos resultados obtidos com o software. 
- [] O paper precisa estar disponibilizado no repositório do Git, na mesma branch que o código fonte.
- [x] Repositório pode ser clonado sem registro. 
- [] Repositório pode ser encontrado através de navegação online sem registro.
- [] Ter um *issue tracker* que possa ser acessado/lido sem registro.
- [] Permitir que pessoas criem issues/file tickets contra o repositório.

## Critérios a serem considerados

- [] Idade do software (se é um projeto bem estabelecido ou se tem um longo histórico de commits).
- [] Número de commits.
- [] Número de autores.
- [] Número de linhas de código (ter mais de 1000 é desejável).
- [x] Se o software já foi citado em papers acadêmicos.
- [] Se o software é útil o suficiente, com chances de ser citado "by your peer group" (não entendi se é o grupo de autores ou o grupo que estará revisando.)
- [] Ser "feature-complete" (não querem soluções pela metade.)
- [] Estar disponibilizado em um pacote que segue o padrão do Python.
- [] Ter um design para receber manutenção (não ser só uma modificação pontual de um software já existente.)

## Melhorando nossas chances (não obrigatório)

- [] Ser modular, bons testes, boa documentação e boa *maintainability*.
- [] Enviar uma contribuição que descreva a aplicação científica do projeto. (Mais detalhes [aqui](https://joss.readthedocs.io/en/latest/submitting.html#co-publication-of-science-methods-and-software)).

## Escrita do paper

- [] Deve ser escrito em MarkDown e compilado com Pandoc (criar uma [GitHub Action](https://joss.readthedocs.io/en/latest/submitting.html#github-action)).
- [] Seguir o [template](https://joss.readthedocs.io/en/latest/submitting.html#example-paper-and-bibliography).
- [] (Opcional) Criar um arquivo de metadados utilizando o [script](https://gist.github.com/arfon/478b2ed49e11f984d6fb).

- Explicar as funcionalidades e domínio de aplicação para um leitor não-especialista.
- Explicar as aplicações do software.
- Paper de 250 a 1000 palavres

### Deve incluir

- [] Lista de autores e suas afiliações.
- [] Um sumário em alto-nivel das funcionalidades do software para um leitor não-especialista (introdução).
- [] "Statement of need": seção para ilustrar claramente o propósito do software e contextualiza o trabalho relacionado.
- [] Lista de referências chave, incluindo a outros softwares endereçando necessidades relacionadas.
- [] Menção de projetos que utilizam o software, de forma a demonstrar como este software oportunizou a realização destes projetos.
- [] Reconhecimento de suporte financeiro (não aplicável a nós)

## Revisão do paper

Revise o paper antes da submissão utilizando os seguintes checklists:

- [] [Checklist dos requisitos](https://joss.readthedocs.io/en/latest/review_checklist.html).
- [] [Checklist dos critérios](https://joss.readthedocs.io/en/latest/review_criteria.html).

## Referências

[Submitting a paper to JOSS](https://joss.readthedocs.io/en/latest/submitting.html)