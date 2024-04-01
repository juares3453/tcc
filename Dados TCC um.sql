WITH Dados AS (
    SELECT
        ISNULL((SELECT TOP 1 F.DsNome FROM GTCFunDp F WHERE F.NrCPF= A.CdMotorista),0) AS [Motorista],
        ISNULL((SELECT TOP 1 '1' FROM SISVeicu X WHERE X.NrPlaca = A.NrPlaca AND X.InSituacaoVeiculo = 1),0) AS [InTipoVeiculo],
        *
    FROM 
        softran_rasador.dbo.TC_HistEntregaFilial A
),
Tabela AS (
    SELECT 
        nrnotafiscal,
        cdromaneio,
        MIN(CONVERT(DATETIME, CONVERT(VARCHAR, CAST(DtMovimentacao AS DATE)) + ' 06:00:00', 120)) AS DataMinima,
        MAX(CAST(DtMovimentacao AS DATETIME)) AS HoraMaxima,
        DATEDIFF(HOUR, 
                 MIN(CONVERT(DATETIME, CONVERT(VARCHAR, CAST(DtMovimentacao AS DATE)) + ' 06:00:00', 120)), 
                 MAX(CAST(DtMovimentacao AS DATETIME))) AS DiferencaEmHoras
    FROM 
        Dados A
    LEFT JOIN 
        softran_rasador.dbo.ESP35303 B ON A.NrPlaca = B.NrPlaca 
    LEFT JOIN 
        softran_rasador.dbo.ESP35302 C ON C.NrCodigoBarras = B.NrCodigoBarras 
    WHERE 
        B.InCategoria = '6'
        AND A.Data = CAST(B.DtMovimentacao AS DATE)
    GROUP BY 
        CdRomaneio,
        NrNotaFiscal
)
SELECT
    DATEPART(day, data) AS Dia,
    DATEPART(month, data) AS Mes,
    DATEPART(year, data) AS Ano,
    A.Filial,
    ISNULL(QtConfLeitorCar,0) as conf_carregamento,
    ISNULL(QtConfLeitorSmart,0) as conf_entrega,
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
FROM 
    softran_rasador.dbo.TC_HistEntregaFilial A
LEFT JOIN 
    Tabela B ON A.CdRomaneio = B.CdRomaneio 
WHERE 
    ISDATE(Hrchegada) = 1
    AND ISDATE(Hrsaida) = 1
    AND KM_C <> 0 
    AND KM_C > 0
ORDER BY 
    Ano, Mes, Dia