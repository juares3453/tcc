DECLARE @VlDiesel2019 numeric(14,2) = 3.55,
        @VlDiesel2020 numeric(14,2) = 3.85,
        @VlDiesel2021 numeric(14,2) = 3.72,
        @VlDiesel2022 numeric(14,2) = 5.41,
        @VlDiesel2023 numeric(14,2) = 6.25,
        @VlPedagio numeric(14,2) = 0.07,
        @VlCusto numeric(14,2),
        @KmMediaComb numeric(14,2) = 3,
        @KmMediaArla numeric(14,2) = 100,
        @Comb2023 numeric(14,2),
        @Comb2022 numeric(14,2),
        @Comb2021 numeric(14,2),
        @Comb2020 numeric(14,2),
        @Comb2019 numeric(14,2),
        @VlArla numeric(14,2) = 4,
        @Arla numeric(14,2),
        @VlManutencao numeric(14,2) = 0.015;

SET @Comb2023 = @VlDiesel2023 / @KmMediaComb;
SET @Comb2022 = @VlDiesel2022 / @KmMediaComb;
SET @Comb2021 = @VlDiesel2021 / @KmMediaComb;
SET @Comb2020 = @VlDiesel2020 / @KmMediaComb;
SET @Comb2019 = @VlDiesel2019 / @KmMediaComb;
SET @Arla = @VlArla / @KmMediaArla;

WITH Dados AS (
    SELECT
        DATEPART(day, A.data) AS Dia,
        DATEPART(month, A.data) AS Mes,
        DATEPART(year, A.data) AS Ano,
		A.CdRomaneio,
		A.CdEmpresa,
        B.NrPlaca,
        C.DsTpVeiculo,
        D.DsModelo,
        B.DsAnoFabricacao,
        CASE
            WHEN ISNULL(A.QtConfLeitorCar, 0) = 0 THEN '0'
            WHEN A.QtConfLeitorCar > 0 THEN A.QtConfLeitorSmart
        END AS conf_carregamento,
        CASE
            WHEN ISNULL(A.QtConfLeitorSmart, 0) = 0 THEN '0'
            WHEN A.QtConfLeitorSmart > 0 THEN A.QtConfLeitorSmart
        END AS conf_entrega,
        DATEDIFF(HOUR, CONVERT(time, A.HrSaida), CONVERT(time, A.HrChegada)) AS tempo_total,
        A.KM_C - A.KM_S AS km_rodado,
        A.NrAuxiliares,
        A.VlCapacVeic,
		FreteEx,
		QtPeso,
		QtPesoEx,
		QtEntregaEx,
		QtEntregas,
		QtVolumeEx,
		Frete,
		QtVolume
    FROM TC_HistEntregaFilial A
    INNER JOIN SISVeicu B ON A.NrPlaca = B.Nrplaca
    LEFT JOIN Sistpvei C ON B.CdTipoVeiculo = C.CdTpVeiculo
    LEFT JOIN SISMdVei D ON B.CdModelo = D.CdModelo
    WHERE ISDATE(A.Hrchegada) = 1 
      AND ISDATE(A.Hrsaida) = 1 
      AND A.KM_C <> 0 
      AND A.KM_C > KM_S
),

Calculos AS (
    SELECT
        D.Dia,
        D.Mes,
        D.Ano,
		D.CdEmpresa,
		D.CdRomaneio,
        D.NrPlaca,
        D.DsTpVeiculo,
        D.DsModelo,
        D.DsAnoFabricacao,
        D.km_rodado,
        D.VlCapacVeic,
        D.NrAuxiliares,
        D.Qtpeso,
        D.QtpesoEx,
        D.QtEntregas,
        D.QtVolume,
        D.QtVolumeEx,
        D.Frete,
        D.FreteEx,
        CASE 
            WHEN D.Ano = 2019 THEN (@Comb2019 * D.KM_Rodado) + (@VlPedagio * D.KM_Rodado) + (@Arla * D.KM_Rodado) + (@VlManutencao * D.KM_Rodado)
            WHEN D.Ano = 2020 THEN (@Comb2020 * D.KM_Rodado) + (@VlPedagio * D.KM_Rodado) + (@Arla * D.KM_Rodado) + (@VlManutencao * D.KM_Rodado)
            WHEN D.Ano = 2021 THEN (@Comb2021 * D.KM_Rodado) + (@VlPedagio * D.KM_Rodado) + (@Arla * D.KM_Rodado) + (@VlManutencao * D.KM_Rodado)
            WHEN D.Ano = 2022 THEN (@Comb2022 * D.KM_Rodado) + (@VlPedagio * D.KM_Rodado) + (@Arla * D.KM_Rodado) + (@VlManutencao * D.KM_Rodado)
            WHEN D.Ano = 2023 THEN (@Comb2023 * D.KM_Rodado) + (@VlPedagio * D.KM_Rodado) + (@Arla * D.KM_Rodado) + (@VlManutencao * D.KM_Rodado)
        END AS VlCusto,
        D.FreteEx,
        CAST((D.Qtpeso / NULLIF(D.VlCapacVeic, 0)) * 100 AS numeric(14,2)) AS [%CapacidadeCarre],
        CAST((D.QtpesoEx / NULLIF(D.VlCapacVeic, 0)) * 100 AS numeric(14,2)) AS [%CapacidadeEntr],
        ISNULL(CAST((D.QtEntregaEx / NULLIF(D.QtEntregas, 0)) * 100 AS numeric(14,2)), 0.00) AS [%Entregas],
        ISNULL(CAST((D.QtVolumeEx / NULLIF(D.QtVolume, 0)) * 100 AS numeric(14,2)), 0.00) AS [%VolumesEntr],
        ISNULL(CAST((D.QtPesoEx / NULLIF(D.QtPeso, 0)) * 100 AS numeric(14,2)), 0.00) AS [%PesoEntr],
        ISNULL(CAST((D.FreteEx / NULLIF(D.Frete, 0)) * 100 AS numeric(14,2)), 0.00) AS [%FreteCobrado]
    FROM Dados D
)
SELECT distinct
    C.Dia,
    C.Mes,
    C.Ano,
    C.DsTpVeiculo,
    C.DsModelo,
    C.DsAnoFabricacao,
    D.Qtpeso,
    D.QtpesoEx,
    D.QtEntregas,
    D.QtVolume,
    D.QtVolumeEx,
    D.Frete,
    D.FreteEx,
    CASE 
        WHEN COALESCE(A.TotalViagem,C.Vlcusto) IS NULL OR A.TotalViagem = 0 THEN C.VlCusto
        ELSE A.TotalViagem
    END AS VlCusto,
    C.km_rodado,
    C.VlCapacVeic,
    C.NrAuxiliares,
    C.[%CapacidadeCarre],
    C.[%CapacidadeEntr],
    C.[%Entregas],
    C.[%VolumesEntr],
    C.[%PesoEntr],
    C.[%FreteCobrado],
    C.FreteEx,
    C.FreteEx - (CASE WHEN A.TotalViagem IS NULL OR A.TotalViagem = 0 THEN C.VlCusto ELSE A.TotalViagem END) AS Lucro,
    ISNULL(CAST((C.FreteEx - (CASE WHEN A.TotalViagem IS NULL OR A.TotalViagem = 0 THEN C.VlCusto ELSE A.TotalViagem END)) / NULLIF(C.FreteEx, 0) * 100 AS numeric(14,2)), 0) AS [%Lucro]
FROM Calculos C
LEFT JOIN ResumoViagem A ON A.CdEmpresa = C.CdEmpresa AND A.CdRomaneio = C.CdRomaneio
ORDER BY C.Ano, C.Mes, C.Dia;
