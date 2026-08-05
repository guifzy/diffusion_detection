# Grupo E - fotometria regional

## Estado metodológico

O grupo foi renomeado de física para fotometria regional na versão 0.2.0. O
prefixo é `photo`, e o notebook utiliza `src.shared.features.group_e`.

## Métricas

Para cada região:

- média, desvio, IQR, contraste robusto e entropia de luminância;
- médias e desvios cromáticos em Lab;
- média e desvio de saturação;
- valor médio em HSV;
- média e desvio do campo de iluminação suavizado;
- energia, dispersão, coerência e direção de seu gradiente;
- razões de pixels escuros, claros e clipados.

O sigma da suavização é relativo ao tamanho facial. A direção média usa
estatística circular ponderada pela magnitude. Comparações angulares usam
distância circular.

## Assimetria

São calculadas assimetrias normalizadas esquerda-direita e topo-base. O
desequilíbrio de quadrantes utiliza quatro quadrantes efetivos.

## Limite de interpretação

As métricas são proxies fotométricos. Elas não estimam fonte de luz, normais,
reflectância completa ou componente especular.

`shadow_ratio` e `highlight_ratio` baseados em percentis foram removidos porque
permaneciam próximos de 20% por construção.

## Candidatos de sombra

A luminância é linearizada e decomposta por uma aproximação Retinex no domínio
logarítmico. Pixels suficientemente abaixo da iluminação mediana da caixa
expandida formam candidatos. O grupo mede razão, profundidade, fronteira,
gradiente e mudança cromática desses candidatos.

Esse resultado é uma evidência física informada, não prova de sombra: materiais
escuros e exposição local ainda podem produzir resposta semelhante. Validação
geométrica 3D permanece futura.

Reflexos oculares e estabilidade temporal estão explicitamente adiados.

O dicionário completo está em `docs/contrato_sinais_v0_2.md`.
