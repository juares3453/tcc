DECLARE @VlDiesel2019 numeric(14,2)
DECLARE @VlDiesel2020 numeric(14,2)
DECLARE @VlDiesel2021 numeric(14,2)
DECLARE @VlDiesel2022 numeric(14,2)
DECLARE @VlDiesel2023 numeric(14,2)
DECLARE @VlPedagio numeric(14,2)
DECLARE @VlCusto numeric(14,2)
DECLARE @KmMediaComb numeric(14,2)
DECLARE @KmMediaArla numeric(14,2)
DECLARE @Comb2023 numeric(14,2)
DECLARE @Comb2022 numeric(14,2)
DECLARE @Comb2021 numeric(14,2)
DECLARE @Comb2020 numeric(14,2)
DECLARE @Comb2019 numeric(14,2)
DECLARE @VlArla numeric(14,2)
DECLARE @Arla numeric(14,2)
DECLARE @VlManutencao numeric(14,2)

SET @VlArla = '4'
SET @KmMediaArla = '100'
SET @KmMediaComb = '3'
SET @VlCusto = '7.10'
SET @VlDiesel2019 = '3.55'
SET @VlDiesel2020 = '3.85'
SET @VlDiesel2021 = '3.72'
SET @VlDiesel2022 = '5.41'
SET @VlDiesel2023 = '6.25'
SET @VlPedagio = '0.07'
SET @VlManutencao = '0.015'

SET @Comb2023 = @VlDiesel2023/@KmMediaComb
SET @Comb2022 = @VlDiesel2022/@KmMediaComb
SET @Comb2021 = @VlDiesel2021/@KmMediaComb
SET @Comb2020 = @VlDiesel2020/@KmMediaComb
SET @Comb2019 = @VlDiesel2019/@KmMediaComb
SET @Arla = @VlArla/@KmMediaArla

DECLARE @tabela TABLE (
dia int,
mes int, 
ano int,
nrplaca varchar(max),
dstpveiculo varchar(max), 
dsmodelo varchar(max),
dsanofabricacao int,
filial varchar(max),
conf_carregamento varchar(max),
conf_entrega varchar(max),
tempo_total int,
km_rodado int,
auxiliares int,
capacidade int,
vlcusto numeric(14,2),
frete_ex numeric(14,2),
capacidadecarre numeric(14,2),
capacidadeentrega numeric(14,2),
entregas numeric(14,2),
volumes_entregas numeric(14,2),
peso_entregas numeric(14,2),
frete_cobrado numeric(14,2)
)

WITH Dados AS(
SELECT
	ISNULL((SELECT TOP 1 F.DsNome FROM GTCFunDp F WHERE F.NrCPF= A.CdMotorista),0) AS [Motorista],
	ISNULL((SELECT TOP 1 '1' FROM SISVeicu X WHERE X.NrPlaca = A.NrPlaca AND X.InSituacaoVeiculo = 1),0) AS [InTipoVeiculo],
	* from TC_HistEntregaFilial A
)

INSERT INTO @tabela
select 
	DATEPART(day, data) AS Dia,
	DATEPART(month, data) AS Mês,
	DATEPART(year, data) AS Ano,
	B.NrPlaca,
	C.DsTpVeiculo,
	D.DsModelo,
	A.DsAnoFabricacao,
	Filial,
	CASE
		WHEN ISNULL(QtConfLeitorCar,0) = 0 THEN '0' 
		WHEN QtConfLeitorCar > 0 THEN '1'
	END AS conf_carregamento,
	CASE
		WHEN ISNULL(QtConfLeitorSmart,0) = 0 THEN '0' 
		WHEN QtConfLeitorSmart > 0 THEN '1'
	END AS conf_entrega,
	DATEDIFF(HOUR, CONVERT(time, HrSaida), CONVERT(time, HrChegada)) as tempo_total,
	KM_C - KM_S as km_rodado,
	NrAuxiliares as auxiliares,
	VlCapacVeic as capacidade,
	CASE 
		WHEN DATEPART(year, data) = '2019' THEN (@Comb2019 * (KM_C - KM_S)) + (@VlPedagio * (KM_C - KM_S)) + (@Arla * (KM_C - KM_S)) + (@VlManutencao * (KM_C - KM_S))
		WHEN DATEPART(year, data) = '2020' THEN (@Comb2020 * (KM_C - KM_S)) + (@VlPedagio * (KM_C - KM_S)) + (@Arla * (KM_C - KM_S)) + (@VlManutencao * (KM_C - KM_S))
		WHEN DATEPART(year, data) = '2021' THEN (@Comb2021 * (KM_C - KM_S)) + (@VlPedagio * (KM_C - KM_S)) + (@Arla * (KM_C - KM_S)) + (@VlManutencao * (KM_C - KM_S))
		WHEN DATEPART(year, data) = '2022' THEN (@Comb2022 * (KM_C - KM_S)) + (@VlPedagio * (KM_C - KM_S)) + (@Arla * (KM_C - KM_S)) + (@VlManutencao * (KM_C - KM_S))
		WHEN DATEPART(year, data) = '2023' THEN (@Comb2023 * (KM_C - KM_S)) + (@VlPedagio * (KM_C - KM_S)) + (@Arla * (KM_C - KM_S)) + (@VlManutencao * (KM_C - KM_S))
	END as VlCusto,
	FreteEx,
	CAST((Qtpeso/NULLIF(VlCapacVeic,0)) * 100 as numeric(14,2)) as [%CapacidadeCarre],
	CAST((QtpesoEx/NULLIF(VlCapacVeic,0)) * 100 as numeric(14,2)) as [%CapacidadeEntr],
	ISNULL(CAST((QtEntregaEx/NULLIF(QtEntregas,0)) * 100 as numeric(14,2)),0.00) as [%Entregas],
	ISNULL(CAST((QtVolumeEx/NULLIF(QtVolume,0)) * 100 as numeric(14,2)),0.00) as [%VolumesEntr],
	ISNULL(CAST((QtPesoEx/NULLIF(QtPeso,0)) * 100 as numeric(14,2)),0.00) as [%PesoEntr],
	ISNULL(CAST((FreteEx/NULLIF(Frete,0)) * 100 as numeric(14,2)),0.00) as [%FreteCobrado]
from Dados B
	inner join SISVeicu A ON A.NrPlaca = B.Nrplaca
	left join Sistpvei C ON A.CdTipoVeiculo = C.CdTpVeiculo
	left join SISMdVei D ON A.CdModelo = D.CdModelo
where ISDATE(Hrchegada) = 1
and ISDATE(Hrsaida) = 1
and KM_C <> 0 
and KM_C > 0
order by Ano, Mês, Dia

Select 
	dia, 
	mes, 
	ano,
	filial, 
	conf_carregamento,
	conf_entrega,
	tempo_total,
	km_rodado,
	auxiliares,
	capacidade,
	vlcusto,
	frete_ex,
	frete_ex-vlcusto as liquido,
	ISNULL(CAST((frete_ex-vlcusto)/NULLIF(frete_ex,0) * 100 as numeric(14,2)),0) as [%Faturado],
	capacidadecarre as [%capacidadecarre],
	capacidadeentrega as [%capacidadeentrega],
	entregas as [%entrega],
	volumes_entregas as [%volumes_entrega],
	peso_entregas as [%peso_entrega],
	frete_cobrado as [%frete_cobrado]
from @tabela