# Grupo A - textura e derivadas espaciais

## Estado metodológico

O grupo está consolidado na versão 0.2.0. O notebook utiliza diretamente
`src.shared.features.group_a`; suas células locais de visualização não definem
as métricas científicas.

## LBP

O LBP uniforme é calculado nas escalas `(P=8,R=1)`, `(P=16,R=2)` e
`(P=24,R=3)`. Cada região produz entropia normalizada, uniformidade e razão de
bins ativos. As regiões são comparadas por contrastes assinados, absolutos e
normalizados, além da distância de Jensen-Shannon entre histogramas.

Média, energia e curtose do código LBP foram removidas porque os códigos são
categorias.

## Sobel

O grupo emite magnitude média, desvio, mediana, percentil 95, energia,
entropia de magnitude, entropia de orientação, coerência angular e razão de
gradientes fortes.

A coerência é a magnitude da média circular ponderada:

\[
C_\theta=
\left|\frac{\sum m e^{i\theta}}{\sum m+\varepsilon}\right|.
\]

A antiga variável `coherence`, que media apenas a proporção acima da média
global, foi substituída por `strong_gradient_ratio`.

## Laplaciano

Energia, variância e curtose usam a resposta assinada. Média, mediana e
percentil 95 de magnitude usam o valor absoluto. O histograma da resposta
padronizada fornece entropia e distância de Jensen-Shannon.

## Saída

O nível por frame é produzido pelo extrator compartilhado. O nível por vídeo
contém média, desvio-padrão e mediana. Não há atributos temporais nesta versão.

O dicionário completo está em `docs/contrato_sinais_v0_2.md`.
