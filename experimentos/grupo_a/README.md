# Metricas atuais do Grupo A

Este arquivo documenta o conjunto de colunas atualmente gerado no pipeline de metricas por frame.

## Colunas de identificacao e controle

- `video_name`
- `frame`
- `label`

## LBP (textura)

- `lbp_face_entropy`
- `lbp_face_uniformity`
- `lbp_border_entropy`
- `lbp_bg_entropy`
- `face_bg_entropy_diff`
- `face_bg_uniformity_diff`
- `face_bg_sparsity_diff`
- `face_bg_hist_dist`
- `face_border_entropy_diff`
- `face_border_uniformity_diff`
- `face_border_sparsity_diff`
- `face_border_hist_dist`
- `border_bg_entropy_diff`
- `border_bg_uniformity_diff`
- `border_bg_sparsity_diff`
- `border_bg_hist_dist`

## Sobel (bordas e coerencia direcional)

- `sobel_face_entropy`
- `sobel_face_coherence`
- `face_bg_coherence_diff`
- `face_bg_energy_diff`
- `face_border_coherence_diff`
- `face_border_energy_diff`
- `border_bg_coherence_diff`
- `border_bg_energy_diff`

## Laplacian (alta frequencia e nitidez)

- `lap_face_energy`
- `lap_face_kurtosis`
- `face_bg_kurtosis_diff`
- `face_border_kurtosis_diff`
- `border_bg_kurtosis_diff`

## Lista completa (ordem atual)

```text
'video_name', 'lbp_face_entropy', 'lbp_face_uniformity',
'lbp_border_entropy', 'lbp_bg_entropy', 'face_bg_entropy_diff',
'face_bg_uniformity_diff', 'face_bg_sparsity_diff', 'face_bg_hist_dist',
'face_border_entropy_diff', 'face_border_uniformity_diff',
'face_border_sparsity_diff', 'face_border_hist_dist',
'border_bg_entropy_diff', 'border_bg_uniformity_diff',
'border_bg_sparsity_diff', 'border_bg_hist_dist', 'sobel_face_entropy',
'sobel_face_coherence', 'face_bg_coherence_diff', 'face_bg_energy_diff',
'face_border_coherence_diff', 'face_border_energy_diff',
'border_bg_coherence_diff', 'border_bg_energy_diff', 'lap_face_energy',
'lap_face_kurtosis', 'face_bg_kurtosis_diff',
'face_border_kurtosis_diff', 'border_bg_kurtosis_diff', 'frame',
'label'
```