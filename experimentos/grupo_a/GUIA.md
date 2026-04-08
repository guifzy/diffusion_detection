# Guia Detalhado de Funcoes - Grupo A

Este documento descreve, em detalhes, cada funcao presente em experimentos_a.py, explicando o objetivo, entradas, saidas e contribuicao para o resultado final (score por familia e exportacao de videos anotados).

## Visao geral do pipeline

O fluxo final do codigo segue esta ordem:

1. Leitura de video e utilitarios visuais.
2. Extracao de sinais por familia:
   - LBP: padroes locais de textura.
   - Sobel: gradiente e direcionalidade de bordas.
   - Laplacian: alta frequencia e variacao de nitidez.
3. Agregacao temporal das metricas de frame para resumo do video.
4. Calculo de score por familia.
5. Processamento orientado por dataframe de metadados.
6. Exportacao de videos em output/lbp, output/sobel e output/laplacian com overlays.

---

## 1) Base de leitura e visualizacao

### load_video_frames(video_path)
- Objetivo: carregar todos os frames de um video para memoria.
- Entrada: caminho do video.
- Saida: array numpy com frames BGR.
- Papel no resultado final: e a base de todo o processamento das familias.

### draw_text_block(img, texts, x=10, y=20)
- Objetivo: desenhar um bloco de texto semitransparente sobre a imagem.
- Entrada: imagem, lista de linhas de texto e posicao inicial.
- Saida: imagem com overlay.
- Papel no resultado final: responsavel por imprimir label, score e metricas importantes nos videos exportados.

### _normalize_to_uint8(array)
- Objetivo: normalizar array numerico para faixa 0-255 em uint8.
- Entrada: array float/int.
- Saida: array uint8.
- Papel no resultado final: padroniza mapas para visualizacao colorida (principalmente Sobel/Laplacian/LBP).

---

## 2) Familia LBP (textura local)

### spatial_entropy(lbp, grid=4, n_bins=10)
- Objetivo: medir entropia local por regioes do mapa LBP.
- Entrada: mapa LBP, tamanho da grade e bins.
- Saida: media e desvio padrao das entropias por patch.
- Papel no resultado final: fornece indicio de variacao espacial da textura.

### compare_region_histograms(hists)
- Objetivo: comparar histogramas de regioes via distancia cosseno.
- Entrada: lista de histogramas.
- Saida: media e desvio das distancias entre pares.
- Papel no resultado final: mede incoerencia de textura entre regioes do frame.

### compute_lbp_visual(gray)
- Objetivo: gerar LBP e mapa colorido para visualizacao.
- Entrada: frame em escala de cinza.
- Saida: mapa LBP numerico e mapa colorido.
- Papel no resultado final: fornece preview visual para exportacao.

### compute_lbp_image(gray)
- Objetivo: extrair metricas LBP de uma imagem.
- Entrada: imagem em cinza.
- Saida: dicionario de metricas + histograma LBP global.
- Papel no resultado final: base de features LBP por frame.

### compute_lbp_frame(frame)
- Objetivo: extrair LBP de frame completo e regioes.
- Entrada: frame BGR.
- Saida: metricas de frame + mapa visual LBP.
- Papel no resultado final: unidade basica de inferencia da familia LBP.

### compute_lbp_metrics_video(sample)
- Objetivo: retornar relatorio de video para LBP via pipeline unificado.
- Entrada: caminho do video.
- Saida: relatorio com summary, score, contribuicoes e frame_metrics.
- Papel no resultado final: interface rapida para score da familia LBP.

### play_video_lbp_image(sample, interval=40, show_players=True)
- Objetivo: visualizar animacao do mapa LBP frame a frame em notebook.
- Entrada: caminho do video, intervalo, controle de exibicao.
- Saida: exibicao HTML + retorno do relatorio LBP.
- Papel no resultado final: depuracao visual e analise qualitativa.

---

## 3) Familia Laplacian (alta frequencia/nitidez)

### compute_laplacian_forensic(gray)
- Objetivo: calcular resposta Laplaciana e mapa colorido.
- Entrada: frame em cinza.
- Saida: magnitude absoluta laplaciana e mapa colorido.
- Papel no resultado final: sinal base para features de nitidez e ruido de alta frequencia.

### laplacian_spatial_features(lap_abs, grid=3)
- Objetivo: medir variacao espacial da energia Laplaciana por patches.
- Entrada: mapa laplaciano absoluto.
- Saida: desvio dos meios e desvio dos desvios por patch.
- Papel no resultado final: captura heterogeneidade espacial.

### projection_features_laplacian(lap_abs, grid=3)
- Objetivo: medir variacao estrutural horizontal/vertical por grade.
- Entrada: mapa laplaciano absoluto.
- Saida: variacao horizontal e vertical.
- Papel no resultado final: indica assimetrias estruturais globais.

### compute_laplacian_image(gray)
- Objetivo: extrair features laplacianas globais da imagem.
- Entrada: imagem em cinza.
- Saida: dicionario de metricas + histograma laplaciano.
- Papel no resultado final: base das metricas Laplacian por regiao.

### laplacian_region_heterogeneity(metrics)
- Objetivo: medir dispersao de lap_var entre regioes.
- Entrada: metricas regionais.
- Saida: desvio e range entre regioes.
- Papel no resultado final: captura inconsistencias regionais de alta frequencia.

### laplacian_region_consistency(region_hists)
- Objetivo: medir consistencia entre histogramas regionais.
- Entrada: histogramas por regiao.
- Saida: media e desvio das distancias cosseno entre pares.
- Papel no resultado final: mede coerencia global entre regioes.

### laplacian_score_frame(metrics)
- Objetivo: sintetizar score laplaciano do frame.
- Entrada: metricas do frame.
- Saida: score escalar.
- Papel no resultado final: score intermediario de frame para interpretacao forense.

### compute_laplacian_frame(frame)
- Objetivo: extrair metricas laplacianas por regiao e score do frame.
- Entrada: frame BGR.
- Saida: metricas do frame + mapa visual.
- Papel no resultado final: unidade de processamento da familia Laplacian.

### compute_laplacian_metrics_video(sample)
- Objetivo: retornar relatorio de video para Laplacian via pipeline unificado.
- Entrada: caminho do video.
- Saida: relatorio com summary, score e contribuicoes.
- Papel no resultado final: interface de score por video para Laplacian.

---

## 4) Familia Sobel (bordas e direcao)

### compute_sobel_forensic(gray)
- Objetivo: calcular gradiente Sobel (magnitude e angulo) e mapa colorido.
- Entrada: frame em cinza.
- Saida: magnitude normalizada, angulo e visual colorido.
- Papel no resultado final: sinal base de bordas/direcoes.

### sobel_spatial_features(mag, grid=3)
- Objetivo: medir variacao espacial da magnitude de gradiente.
- Entrada: magnitude Sobel.
- Saida: dispersao dos meios e dos desvios por patch.
- Papel no resultado final: detecta instabilidade local de bordas.

### directional_coherence(angle, grid=3)
- Objetivo: quantificar coerencia de orientacoes de gradiente entre patches.
- Entrada: angulos Sobel.
- Saida: escalar de coerencia direcional.
- Papel no resultado final: detecta inconsistencias de orientacao.

### projection_features(mag, grid=3)
- Objetivo: medir estrutura horizontal e vertical do gradiente.
- Entrada: magnitude Sobel.
- Saida: variacao horizontal e vertical.
- Papel no resultado final: captura desequilibrios estruturais.

### compute_sobel_image(gray)
- Objetivo: extrair metricas Sobel da imagem.
- Entrada: frame em cinza.
- Saida: metricas, histograma angular e mapa visual.
- Papel no resultado final: base das features Sobel de frame.

### region_heterogeneity(metrics)
- Objetivo: medir dispersao de grad_std entre regioes.
- Entrada: metricas regionais.
- Saida: desvio e range regionais.
- Papel no resultado final: componente de incoerencia local.

### directional_asymmetry(metrics)
- Objetivo: medir diferenca de coerencia entre pares opostos (top-bottom, left-right).
- Entrada: metricas regionais.
- Saida: diferencas tb e lr.
- Papel no resultado final: componente de assimetria direcional.

### angular_consistency(region_angle_hists)
- Objetivo: medir consistencia angular entre regioes via distancia cosseno.
- Entrada: histogramas angulares por regiao.
- Saida: media e desvio das distancias.
- Papel no resultado final: componente de coerencia angular global.

### sobel_score_frame(metrics)
- Objetivo: consolidar heterogeneidade, assimetria e consistencia em score de frame.
- Entrada: metricas do frame.
- Saida: score escalar.
- Papel no resultado final: score intermediario de frame para interpretacao.

### compute_sobel_frame(frame)
- Objetivo: extrair metricas Sobel por regiao e score do frame.
- Entrada: frame BGR.
- Saida: metricas do frame + mapa Sobel colorido.
- Papel no resultado final: unidade de processamento Sobel.

### compute_sobel_scores(sample)
- Objetivo: retornar relatorio de video para Sobel via pipeline unificado.
- Entrada: caminho do video.
- Saida: relatorio com summary, score e contribuicoes.
- Papel no resultado final: interface de score por video para Sobel.

### play_video_sobel_image(sample, interval=40, show_players=True)
- Objetivo: visualizar mapa Sobel com metricas por frame em notebook.
- Entrada: caminho do video e parametros de exibicao.
- Saida: exibicao HTML + relatorio Sobel.
- Papel no resultado final: apoio de diagnostico visual.

---

## 5) Agregacao e score por familia

### _finite_values(values)
- Objetivo: filtrar valores validos (nao nulos e finitos).
- Entrada: lista de valores.
- Saida: array numerico filtrado.
- Papel no resultado final: evita contaminar agregacao com NaN/inf.

### aggregate_metric_series(frame_metrics, metric_names, reducer=np.median)
- Objetivo: agregar metricas no tempo (frame -> video).
- Entrada: lista de metricas por frame e nomes desejados.
- Saida: resumo por metrica (mediana por padrao).
- Papel no resultado final: converte dinamica temporal em vetor estavel por video.

### _bounded_positive(value)
- Objetivo: transformar valor positivo em escala limitada usando exponencial.
- Entrada: valor escalar.
- Saida: valor no intervalo aproximado [0, 1).
- Papel no resultado final: estabiliza contribuicoes de metricas positivas.

### _inverse_log(value)
- Objetivo: reduzir escala de metricas grandes com funcao inversa logaritmica.
- Entrada: valor escalar.
- Saida: valor comprimido.
- Papel no resultado final: reduz dominancia de magnitudes muito altas.

### score_lbp_summary(summary)
- Objetivo: calcular score final LBP a partir do resumo agregado.
- Entrada: resumo LBP do video.
- Saida: score, componentes e contribuicoes.
- Papel no resultado final: score oficial da familia LBP.

### score_sobel_summary(summary)
- Objetivo: calcular score final Sobel por blocos (basic, spatial, coherence).
- Entrada: resumo Sobel do video.
- Saida: score, componentes e contribuicoes.
- Papel no resultado final: score oficial da familia Sobel.

### score_laplacian_summary(summary)
- Objetivo: calcular score final Laplacian por blocos.
- Entrada: resumo Laplacian do video.
- Saida: score, componentes e contribuicoes.
- Papel no resultado final: score oficial da familia Laplacian.

### collect_family_outputs(sample, frame_fn, include_previews=False)
- Objetivo: executar uma familia frame a frame, coletando metricas e previews.
- Entrada: caminho do video, funcao de frame e flag de previews.
- Saida: frame_metrics e preview_frames.
- Papel no resultado final: camada comum de coleta para todas as familias.

### compute_family_video_report(sample, family_name, include_previews=False)
- Objetivo: pipeline unificado por familia (coleta -> agregacao -> score).
- Entrada: caminho do video e nome da familia.
- Saida: summary, score, componentes, contribuicoes, frame_metrics e previews.
- Papel no resultado final: funcao central de processamento por familia.

---

## 6) Utilitarios de exportacao

### _safe_video_stem(filename)
- Objetivo: sanitizar nome de arquivo para gerar nomes seguros de saida.
- Entrada: nome original.
- Saida: stem limpo.
- Papel no resultado final: evita falhas de IO por caracteres invalidos.

### _get_video_fps(video_path, default_fps=25.0)
- Objetivo: obter FPS real do video, com fallback.
- Entrada: caminho do video.
- Saida: fps float.
- Papel no resultado final: preserva fluidez temporal na exportacao.

### _video_writer_fourcc()
- Objetivo: obter codec mp4v de forma compativel para VideoWriter.
- Entrada: nenhuma.
- Saida: fourcc inteiro.
- Papel no resultado final: garante criacao dos videos de saida.

### _build_overlay_lines(family_name, family_report, label, filename)
- Objetivo: montar linhas de overlay para o video exportado.
- Entrada: familia, relatorio, label e nome do arquivo.
- Saida: lista de strings.
- Papel no resultado final: define exatamente o que aparece no video (label, score e metricas importantes).

### _export_annotated_family_video(preview_frames, output_path, fps, overlay_lines)
- Objetivo: gravar video anotado quadro a quadro.
- Entrada: frames de preview, caminho final, fps e linhas do overlay.
- Saida: booleano indicando sucesso.
- Papel no resultado final: gera o artefato visual final por familia.

---

## 7) Orquestracao final por dataframe

### _select_rows_from_metadata(metadata_df, n_real, n_fake)
- Objetivo: selecionar quantos reais e quantos fakes serao processados.
- Entrada: dataframe de metadados e quantidades desejadas.
- Saida: lista de pares (row, label).
- Papel no resultado final: controla o recorte de videos processados.

### process_group_a_from_metadata(metadata_df, n_real=3, n_fake=3, output_dir="output")
- Objetivo: funcao principal final do fluxo.
- Entrada:
  - metadata_df com colunas Video Ground Truth, Filename e video_path.
  - n_real e n_fake para quantidade de videos de cada classe.
  - output_dir para pasta de saida.
- Saida:
  - results: lista com label, family_scores e output_videos por video processado.
  - output_dir: caminho base da saida.
- Papel no resultado final:
  - calcula score para cada familia;
  - cria pastas por familia;
  - exporta videos anotados por familia;
  - entrega estrutura final simplificada para consumo.

---

## Estrutura final de saida esperada

Para cada video processado em results:
- label: Real ou Fake.
- family_scores:
  - lbp
  - sobel
  - laplacian
- output_videos:
  - caminho do video LBP exportado
  - caminho do video Sobel exportado
  - caminho do video Laplacian exportado

---

## Intencao metodologica final

- LBP captura padroes de textura local e consistencia entre regioes.
- Sobel captura estrutura de bordas, orientacao e coerencia angular.
- Laplacian captura energia de alta frequencia e estabilidade espacial.

A combinacao das tres familias permite analisar o video por sinais complementares, reduzindo dependencia de uma unica pista forense.
