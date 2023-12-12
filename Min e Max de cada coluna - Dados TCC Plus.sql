DECLARE @VlDiesel numeric(14,2)
DECLARE @VlPedagio numeric(14,2)

WITH Dados AS(
SELECT
	ISNULL((SELECT TOP 1 F.DsNome FROM GTCFunDp F WHERE F.NrCPF= A.CdMotorista),0) AS [Motorista],
	ISNULL((SELECT TOP 1 '1' FROM SISVeicu X WHERE X.NrPlaca = A.NrPlaca AND X.InSituacaoVeiculo = 1),0) AS [InTipoVeiculo],
	* from TC_HistEntregaFilial A
)

select 
	MIN(DATEPART(day, data)) AS Dia,
	MAX(DATEPART(day, data)) AS Dia,
	MIN(DATEPART(month, data)) AS Mês,
	MAX(DATEPART(month, data)) AS Mês,
	MIN(DATEPART(year, data)) AS Ano,
	MAX(DATEPART(year, data)) AS Ano,
	MIN(Filial),
	MAX(Filial),
	--CASE
	--	WHEN ISNULL(QtConfLeitorCar,0) = 0 THEN '0' 
	--	WHEN QtConfLeitorCar > 0 THEN '1'
	--END AS conf_carregamento,
	--CASE
	--	WHEN ISNULL(QtConfLeitorSmart,0) = 0 THEN '0' 
	--	WHEN QtConfLeitorSmart > 0 THEN '1'
	--END AS conf_entrega,
	MIN(DATEDIFF(HOUR, CONVERT(time, HrSaida), CONVERT(time, HrChegada))) as tempo_total,
	MAX(DATEDIFF(HOUR, CONVERT(time, HrSaida), CONVERT(time, HrChegada))) as tempo_total,
	MIN(KM_C - KM_S) as km_rodado,
	MAX(KM_C - KM_S) as km_rodado,
	MIN(NrAuxiliares) as auxiliares,
	MAX(NrAuxiliares) as auxiliares,
	MIN(VlCapacVeic) as capacidade,
	MAX(VlCapacVeic) as capacidade,
	MIN(CAST((Qtpeso/NULLIF(VlCapacVeic,0)) * 100 as numeric(14,2))) as [%CapacidadeCarre],
	MAX(CAST((Qtpeso/NULLIF(VlCapacVeic,0)) * 100 as numeric(14,2))) as [%CapacidadeCarre],
	MIN(CAST((QtpesoEx/NULLIF(VlCapacVeic,0)) * 100 as numeric(14,2))) as [%CapacidadeEntr],
	MAX(CAST((QtpesoEx/NULLIF(VlCapacVeic,0)) * 100 as numeric(14,2))) as [%CapacidadeEntr],
	MIN(ISNULL(CAST((QtEntregaEx/NULLIF(QtEntregas,0)) * 100 as numeric(14,2)),0.00)) as [%Entregas],
	MAX(ISNULL(CAST((QtEntregaEx/NULLIF(QtEntregas,0)) * 100 as numeric(14,2)),0.00))  as [%Entregas],
	MIN(ISNULL(CAST((QtVolumeEx/NULLIF(QtVolume,0)) * 100 as numeric(14,2)),0.00)) as [%VolumesEntr],
	MAX(ISNULL(CAST((QtVolumeEx/NULLIF(QtVolume,0)) * 100 as numeric(14,2)),0.00)) as [%VolumesEntr],
	MIN(ISNULL(CAST((QtPesoEx/NULLIF(QtPeso,0)) * 100 as numeric(14,2)),0.00)) as [%PesoEntr],
	MAX(ISNULL(CAST((QtPesoEx/NULLIF(QtPeso,0)) * 100 as numeric(14,2)),0.00)) as [%PesoEntr],
	MIN(ISNULL(CAST((FreteEx/NULLIF(Frete,0)) * 100 as numeric(14,2)),0.00)) as [%FreteCobrado],
	MAX(ISNULL(CAST((FreteEx/NULLIF(Frete,0)) * 100 as numeric(14,2)),0.00)) as [%FreteCobrado]
from Dados B
	inner join SISVeicu A ON A.NrPlaca = B.Nrplaca
	left join Sistpvei C ON A.CdTipoVeiculo = C.CdTpVeiculo
	left join SISMdVei D ON A.CdModelo = D.CdModelo
where ISDATE(Hrchegada) = 1
and ISDATE(Hrsaida) = 1
and KM_C <> 0 
and KM_C > 0