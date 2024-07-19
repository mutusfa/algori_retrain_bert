SELECT
  OcrValueId,
  OcrValue,
  OcrValueId_Correct,
  Correct_Desc AS OcrValue_Correct
FROM [dbo].[Fuzzy_Mappings]
WHERE OcrValueId_Correct IS NOT NULL