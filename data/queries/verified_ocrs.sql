WITH UnclassifiedOcrs AS (
    SELECT [OcrValueId]
          ,[MappingGroupId]
    FROM [dbo].[ClassifiedOcrs] co
    WHERE LastUserUpdated IS NULL
)

SELECT 
       A.[MappingGroupId]
      ,A.[Verified_OcrValueId]
      ,A.[OcrValue_Verified]
      ,A.[Category_AECOC]
      ,A.[Category_Info_Source]
      ,B.OcrValueId
      ,SUM(A.[Receipt_Count]) AS Sum_Receipts
FROM [ssas].[Sub_OCRs] AS A
LEFT JOIN UnclassifiedOcrs AS B
ON A.MappingGroupId = B.MappingGroupId AND A.Verified_OcrValueId = B.OcrValueId
WHERE
    (A.Category_Info_Source <> 'OCRClassification' OR B.OcrValueId IS NULL)
    AND A.Category_AECOC IS NOT NULL
    AND A.Category_Info_Source IS NOT NULL
GROUP BY 
       A.[MappingGroupId]
      ,A.[Verified_OcrValueId]
      ,A.[OcrValue_Verified]
      ,A.[Category_AECOC]
      ,A.[Category_Info_Source]
      ,B.OcrValueId
      ,A.MappingGroupId
