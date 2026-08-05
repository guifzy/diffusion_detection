# Grupo B - estrutura local

## Estado metodológico

O grupo está consolidado na versão 0.2.0. O notebook importa os extratores de
`src.shared.features.group_b`.

## SIFT

Para cada região são calculados:

- quantidade e densidade de keypoints;
- cobertura espacial em grade 4x4;
- média e dispersão de resposta e escala;
- entropia e coerência de orientação;
- entropia e autossimilaridade dos descritores.

A média das distâncias entre todos os descritores de face e fundo foi removida.
Ela comparava conteúdos semanticamente distintos sem matching geométrico.
Matching SIFT entre frames pertence à futura etapa temporal.

## Similaridade de patches

O patch possui tamanho relativo à face, limitado entre 8 e 32 pixels. A
amostragem é determinística e distribuída por toda a região, sem viés para os
primeiros patches em ordem raster.

São produzidas média, desvio, mediana e percentil 95 da similaridade cosseno.
Quantidade de candidatos, quantidade amostrada e cobertura são controles de
qualidade; não recebem contrastes forenses entre regiões.

## Saída

Contrastes regionais possuem versões assinada, absoluta e normalizada. O nível
por vídeo contém média, desvio-padrão e mediana, sem temporalidade ordenada.

O dicionário completo está em `docs/contrato_sinais_v0_2.md`.
