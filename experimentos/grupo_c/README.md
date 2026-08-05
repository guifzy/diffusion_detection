# Grupo C - resíduo bilateral

## Estado metodológico

O grupo está consolidado na versão 0.2.0. O prefixo das colunas é `residual`, e
o notebook utiliza `src.shared.features.group_c`.

## Definição

\[
R=I-\operatorname{Bilateral}(I).
\]

Esse sinal é um baseline de alta passagem. Ele contém ruído, textura, bordas,
sharpening e compressão; portanto não é identificado como PRNU, Noiseprint ou
ruído de sensor.

## Métricas

Para cada região:

- média assinada, desvio, RMS, MAD e percentil 95 absoluto;
- entropia e curtose;
- autocorrelação lag-1 horizontal e vertical;
- correlação média entre canais do resíduo colorido.

`variance` foi removida porque é exatamente `std²`. Os contrastes regionais são
assinados, absolutos e normalizados.

## Limite de interpretação

O sinal deve permanecer como baseline. A inclusão futura de SRM, wavelets ou
Noiseprint deverá usar nomes e versão próprios.

Não há coerência temporal do resíduo nesta versão.

O dicionário completo está em `docs/contrato_sinais_v0_2.md`.
