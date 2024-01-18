SELECT 
 A.CdEmpresaResp AS [Resp],
 case
  when (A.CdEmpresaResp = 1 OR A.CdEmpresaResp = 5 OR A.CdEmpresaResp = 6 and A.CdEmpresaResp = 9) Then 'BG' 
  when (A.CdEmpresaResp = 4 OR A.CdEmpresaResp = 7) Then 'BA' 
  when A.CdEmpresaResp = 3 Then 'CB'
  when A.CdEmpresaResp = 2 Then 'PA'
  when A.CdEmpresaResp = 12 Then 'SJ'
  when A.CdEmpresaResp = 13 Then 'BC'
  when A.CdEmpresaResp = 14 Then 'LD'
  when A.CdEmpresaResp = 16 Then 'EX'
 else '1'  
 end as [CD],
 CLI.CdInscricao AS [CLIENTE],
 substring(CLI.DsEntidade,1,15) as [DS_CLIENTE],
 convert(varchar(5),Month(A.DtBaixa))+'/'+convert(varchar(5), year(A.DtBaixa)) as [Compet.],
 G.DtEmissao as dtcte,
 A.DtEmissao, 
 A.DtOcorrencia,
 A.DtBaixa,
 K.DsLocal,
 CAST(case 
  when D.DSOCORRENCIA like 'FALTA PARCIAL'  Then 'FALTA' 
  when D.DSOCORRENCIA like 'FALTA TOTAL'  Then 'FALTA' 
  when D.DSOCORRENCIA like 'AVARIA TOTAL'  Then 'AVARIA' 
  when D.DSOCORRENCIA like 'AVARIA PARCIAL'  Then 'AVARIA' 
 end as varchar) as [TIPO ACOR],
 CAST(case 
  when year(A.DtBaixa) IS null  Then 'ABERTO' 
  when year(A.DtBaixa) IS NOT null Then 'FECHADO' 
  else '1' 
 end As varchar) as [Situacao],
 A.NrBo,
 CAST(A.DsOcorrencia as varchar) as dsocorrencia,
 SUM(L.VlCusto) as VlCusto
FROM CCECTRME A  (NOLOCK)
LEFT JOIN CCECTRCO C (NOLOCK) ON A.CDEMPRESA = C.CDEMPRESA AND A.NRBO = C.NRBO
LEFT JOIN CCECTRCU L (NOLOCK) ON A.CDEMPRESA = L.CDEMPRESA AND A.NRBO = L.NRBO
LEFT JOIN GTCCONHE G (NOLOCK) ON C.CDEMPRESACONHE = G.CDEMPRESA AND C.NRSEQCONTROLE = G.NRSEQCONTROLE
LEFT JOIN SISCLI CLI (NOLOCK) ON CLI.CdInscricao=G.CdInscricao
LEFT JOIN CCETPOCO D (NOLOCK) ON A.CDOCORRENCIA = D.CDOCORRENCIA
Left Join CCETPLOC K (NOLOCK) ON A.CdLocal = K.CdLocal
WHERE ISNULL(G.INCONHECIMENTO,0)=0 
And A.CdOcorrencia <> '12'
and A.DtEmissao >= '20230101'
and A.DtEmissao <= '20231231'
AND L.VlCusto > 0
and A.NrBo = 9733
and A.CdEmpresa = 4
and substring(CLI.CdInscricao,1,8) not in ('88081039')
GROUP BY A.CdEmpresaResp, A.DtBaixa,  A.DtEmissao, K.DsLocal,
D.DSOCORRENCIA, A.NrBo, CLI.CdInscricao, DsEntidade, DtOcorrencia, g.DtEmissao,  CAST(A.DsOcorrencia as varchar)



