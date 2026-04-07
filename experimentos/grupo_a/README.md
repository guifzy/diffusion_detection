# Metricas utilizadas

## LBP

- `full_entropy`: entropia do histograma LBP do frame inteiro; mede a complexidade global da textura local.
- `full_entropy_spatial_std`: desvio-padrao espacial da entropia LBP; mede a variacao local da textura dentro do frame.
- `lbp_region_distance_mean`: media das distancias entre histogramas LBP das regioes do frame; mede inconsistencias espaciais.
- `lbp_region_distance_std`: desvio-padrao dessas distancias; destaca inconsistencia localizada entre regioes.

As metricas `uniformity` e as entropias por regiao nao entram mais no conjunto final porque repetiam informacao da entropia global e das distancias entre regioes.

## Sobel

- `sobel_region_std`: dispersao regional da intensidade de gradiente; mede heterogeneidade local.
- `sobel_region_range`: amplitude entre regioes; enfatiza extremos locais de gradiente.
- `sobel_tb_coherence_diff`: diferenca de coerencia direcional entre metade superior e inferior.
- `sobel_lr_coherence_diff`: diferenca de coerencia direcional entre metade esquerda e direita.
- `sobel_angle_consistency_mean`: distancia angular media entre regioes; mede incoerencia direcional global.
- `sobel_angle_consistency_std`: dispersao dessas distancias; mede variabilidade da coerencia angular.

As distancias angulares par-a-par especificas nao entram no score final porque repetem a informacao ja resumida pela coerencia angular agregada.

## Laplacian

- `full_frame_lap_var`: variancia do Laplaciano no frame inteiro; mede alta frequencia e nitidez.
- `full_frame_lap_mean_abs`: media absoluta do Laplaciano; reforca a presenca de bordas finas e detalhe local.
- `full_frame_lap_p90`: cauda alta da distribuicao Laplacian; destaca picos de detalhe e acentuacao.
- `full_frame_lap_spatial_mean_std`: variacao espacial da media do Laplaciano; mede heterogeneidade entre regioes.
- `full_frame_lap_spatial_std_std`: variacao espacial da dispersao do Laplaciano; reforca inconsistencias regionais.
- `lap_region_std`: dispersao da variancia Laplacian entre regioes do frame.
- `lap_region_range`: amplitude entre regioes; destaca contraste extremo de nitidez.
- `lap_region_consistency_mean`: distancia media entre histogramas Laplacian das regioes.
- `lap_region_consistency_std`: dispersao dessas distancias.
- `laplacian_score`: score de frame do Laplacian, combinando base, espacialidade e coerencia entre regioes.
- `laplacian_mean`: media temporal de `laplacian_score` ao longo do video.
- `laplacian_var_temporal`: variancia temporal de `laplacian_score`.
- `laplacian_mean_delta`: media da variacao absoluta de `laplacian_score` entre frames consecutivos.

O Laplacian faz sentido como sinal complementar ao Sobel, mas nao substitui completamente o Sobel. Ele descreve segunda derivada, alta frequencia e nitidez, enquanto o Sobel descreve orientacao e coerencia de bordas.

## Temporalidade

Para o Grupo A, a temporalidade mais util e a agregacao simples dos sinais por video: media, desvio e variacao frame a frame.

Nao vale a pena criar um bloco temporal grande antes de validar se os sinais estaticos realmente separam real e fake. Se houver flicker ou instabilidade visual, ai sim a variacao temporal passa a ser informativa; caso contrario, vira ruido.

## Leitura forense

- `LBP` fica para textura e incoerencia espacial fina.
- `Sobel` fica para bordas, gradientes e incoerencia direcional.
- `Laplacian` complementa com nitidez e alta frequencia.

O objetivo e evitar contagem dupla de sinais que descrevem a mesma propriedade visual.

## Benchmark e decisao

- `collect_group_a_video(...)`: coleta as saidas por familia de um video.
- `run_group_a_benchmark(metadata_true, metadata_fake, n_true=5, n_fake=5, max_frames=120, include_previews=False)`: roda o benchmark em lote para qualquer quantidade de videos, sem calibracao previa.
- `summary`: resumo comparativo de reais e falsos com medianas dos scores por familia.

Regra provisoria de decisao:

- `suspeito`: pelo menos duas familias com score alto de anomalia.
- `coerente`: todas as familias com score baixo e pouca dispersao entre elas.
- `inconclusivo`: qualquer caso intermediario.

Os scores por familia sao provisórios e servem para inspecao antes da calibracao formal. Eles usam criterios diferentes por familia: LBP enfatiza textura e incoerencia espacial, Sobel enfatiza bordas, gradientes e direcoes, e Laplacian enfatiza nitidez, alta frequencia e irregularidade entre regioes.