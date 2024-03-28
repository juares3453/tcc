WITH Dados AS(
SELECT
	ISNULL((SELECT TOP 1 F.DsNome FROM GTCFunDp F WHERE F.NrCPF= A.CdMotorista),0) AS [Motorista],
	ISNULL((SELECT TOP 1 '1' FROM SISVeicu X WHERE X.NrPlaca = A.NrPlaca AND X.InSituacaoVeiculo = 1),0) AS [InTipoVeiculo],
	* from TC_HistEntregaFilial A
)

select 
	DATEPART(day, data) AS Dia,
	DATEPART(month, data) AS Mes,
	DATEPART(year, data) AS Ano,
	Filial,
	CASE
		WHEN ISNULL(QtConfLeitorCar,0) = 0 THEN '0' 
		WHEN QtConfLeitorCar > 0 THEN QtConfLeitorCar
	END AS conf_carregamento,
	CASE
		WHEN ISNULL(QtConfLeitorSmart,0) = 0 THEN '0' 
		WHEN QtConfLeitorSmart > 0 THEN QtConfLeitorSmart
	END AS conf_entrega,
	DATEDIFF(HOUR, CONVERT(time, HrSaida), CONVERT(time, HrChegada)) as tempo_total,
	KM_C - KM_S as km_rodado,
	NrAuxiliares as auxiliares,
	VlCapacVeic as capacidade,
	QtEntregas as entregas_total,
	QtEntregaEx as entregas_realizadas,
	QtVolume as volumes_total,
	QtVolumeEx as volumes_entregues,
	QtPeso as peso_total,
	QtPesoEx as peso_entregue,
	Frete as frete_total,
	FreteEx as frete_entregue
from Dados B
where ISDATE(Hrchegada) = 1
and ISDATE(Hrsaida) = 1
and KM_C <> 0 
and KM_C > 0
order by Ano, Mes, Dia
