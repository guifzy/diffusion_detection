# Metricas utilizadas

## LBP

- `lbp_entropy`: entropia do histograma LBP de um frame; mede complexidade/desordem de textura local.
- `lbp_mean`: media da entropia LBP ao longo dos frames do video.
- `lbp_var_temporal`: variancia temporal da entropia LBP entre frames.
- `lbp_mean_delta`: media da variacao absoluta de entropia LBP entre frames consecutivos.
- `lbp_region_diff`: diferenca media absoluta de entropia LBP entre metade esquerda e direita do frame.

## Sobel

- `sobel_mean`: media da magnitude de gradiente (Sobel) no frame; mede intensidade media de bordas.
- `sobel_var_temporal`: variancia temporal da magnitude media Sobel entre frames.
- `sobel_mean_delta`: media da variacao absoluta da magnitude media Sobel entre frames consecutivos.
- `sobel_region_diff`: diferenca media absoluta da magnitude Sobel entre metade esquerda e direita do frame.

## Laplacian

- `laplacian_mean`: media da variancia do Laplaciano ao longo dos frames; mede nivel medio de alta frequencia/foco.
- `laplacian_var_temporal`: variancia temporal da variancia do Laplaciano entre frames.
- `laplacian_mean_delta`: media da variacao absoluta da variancia do Laplaciano entre frames consecutivos.

## Metricas de anomalias temporais

- `lbp_delta`: variacao absoluta da entropia LBP em relacao ao frame anterior.
- `lbp_region_diff`: assimetria esquerda-direita de textura LBP no frame.
- `lbp_temporal_var`: variancia local (janela) da serie LBP para medir instabilidade recente.
- `sobel_delta`: variacao absoluta da magnitude Sobel em relacao ao frame anterior.
- `sobel_region_diff`: assimetria esquerda-direita da magnitude Sobel no frame.
- `sobel_temporal_var`: variancia local (janela) da serie Sobel para medir instabilidade recente.
- `lap_delta`: variacao absoluta da variancia Laplaciana em relacao ao frame anterior.
- `lap_temporal_var`: variancia local (janela) da serie Laplaciana para medir instabilidade recente.
- `score_components`: contribuicoes por metrica (apos normalizacao robusta) usadas no score final.
- `frame_scores` / `instability_score`: score de instabilidade por frame (combinacao ponderada das metricas temporais).
- `score_threshold_p95`: limiar adaptativo de score definido pelo percentil 95 do periodo de warmup.
- `peak_z`: z-score de pico do score atual contra scores recentes; mede anomalia abrupta.
- `frame_alerts`: flag de alerta por frame indicando instabilidade detectada.
- `alert_rate`: proporcao de frames em alerta no video.
- `n_alerts`: quantidade total de frames em alerta no video.
- `score_mean`: media do score de instabilidade no video.
- `score_std`: desvio padrao do score de instabilidade no video.
- `metric_thresholds`: limiares por metrica (percentil 95 no warmup) usados para contagem de votos.

## Metricas de validacao estatistica (secao de validacao formal)

- `AUC-ROC`: area sob a curva ROC; mede separacao global entre classes em todos os limiares.
- `sensibilidade`: taxa de verdadeiros positivos (recall da classe positiva).
- `especificidade`: taxa de verdadeiros negativos.
- `intervalo de confianca (bootstrap)`: faixa de incerteza das metricas estimada por reamostragem.
- `indice de Youden`: criterio para escolher limiar otimo maximizando sensibilidade + especificidade - 1.
