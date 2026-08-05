# Grupo D - frequência espacial

## Estado metodológico

O grupo está consolidado na versão 0.2.0. O notebook utiliza
`src.shared.features.group_d`.

## Preparação espectral

Cada patch é redimensionado para 128x128, centralizado e multiplicado por uma
janela Hann bidimensional:

\[
\tilde I=(I-\bar I)w,\qquad
F=\operatorname{fftshift}(\mathcal{F}_2\{\tilde I\}),\qquad
P=|F|^2.
\]

Todos os patches válidos do contorno e do contexto são analisados. A seleção
anterior de apenas um patch foi removida.

## Métricas

- média e desvio da log-amplitude;
- razões de potência baixa, média e alta;
- centroide radial;
- entropia e planicidade espectral;
- anisotropia angular;
- inclinação do perfil espectral;
- quantidade de patches como controle de qualidade.

As razões são calculadas sobre potência, não sobre log-amplitude. A
`mean_intensity` removida era aproximadamente zero após centralização.

## Limite de interpretação

O domínio espectral é complementar e sensível a gerador, resize e compressão.
Sua robustez deve ser medida antes da seleção para modelagem.

Não há frequência temporal nesta versão.

O dicionário completo está em `docs/contrato_sinais_v0_2.md`.
