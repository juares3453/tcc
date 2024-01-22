SELECT 
 A.CdEmpresaResp AS [Resp],
 CLI.CdInscricao AS [CLIENTE],
 convert(varchar(5),Month(A.DtBaixa))+'/'+convert(varchar(5), year(A.DtBaixa)) as [Compet],
 day(G.DtEmissao) as dtcte,
 month(G.DtEmissao) as mescte,
 year(G.DtEmissao) as anocte,
 day(A.DtEmissao) as dtemissao, 
 month(A.DtEmissao) as mesemissao, 
 year(A.DtEmissao) as anoemissao, 
 day(A.DtOcorrencia) as dtocor,
 month(A.DtOcorrencia) as mesocor,
 year(A.DtOcorrencia) as anoocor,
 day(A.DtBaixa) as dtbaixa,
 month(A.DtBaixa) as mesbaixa,
 year(A.DtBaixa) as anobaixa,
 datediff(DAY,  A.DtEmissao, A.DtOcorrencia) as diasemissao,
 datediff(DAY, A.DtOcorrencia, A.DtBaixa) as diasresolucao,
 K.DsLocal,
 CAST(case 
  when D.DSOCORRENCIA like 'FALTA PARCIAL'  Then 'FALTA' 
  when D.DSOCORRENCIA like 'FALTA TOTAL'  Then 'FALTA' 
  when D.DSOCORRENCIA like 'AVARIA TOTAL'  Then 'AVARIA' 
  when D.DSOCORRENCIA like 'AVARIA PARCIAL'  Then 'AVARIA' 
 end as varchar) as tp_ocor,
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



