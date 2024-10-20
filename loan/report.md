# Loan Agreement Section Extraction: Proof of Concept Report

## 1. Executive Summary

This report presents a proof of concept for extracting specific sections from a large loan agreement document (300,000+ words) using a combination of BM25 search, Large Language Model (LLM) based techniques, and Knowledge Graph (KG) relationships. The goal was to accurately identify and highlight relevant text for various sections of the loan agreement, such as financial covenants, interest and fees, and tranche information. Our hybrid approach demonstrated promising results, effectively combining the strengths of traditional information retrieval methods with advanced natural language processing capabilities and structured knowledge representation.

## 2. Methodology and Experiments

Our approach consisted of the following key steps:

1. **Document Chunking**: The loan agreement PDF was converted to markdown format and split into manageable chunks while preserving semantic integrity.

2. **Keyword Generation**: An LLM was used to generate relevant keywords and phrases for each section based on provided guidelines.

3. **BM25 Search**: We applied BM25 search using the generated keywords to identify relevant chunks for each section. BM25, which stands for Best Matching 25, is a ranking function used in information retrieval. It uses a probabilistic model to score the relevance of documents to a given search query. BM25 takes into account term frequency, inverse document frequency, and document length normalization to provide effective keyword-based searching.

4. **LLM-based Chunk Identification**: An LLM was employed to analyze the document chunks and identify those relevant to each section based on given guidelines.

5. **Knowledge Graph Integration**: Relationships extracted from a Knowledge Graph were added as inputs to the LLM prompts, providing additional context and structure to the analysis.

6. **Result Combination**: The results from BM25, LLM-based searches, and KG-enhanced analysis were combined to produce a final list of relevant chunk indexes for each section.

7. **Highlighting**: Relevant text was highlighted in the original document based on the identified chunk indexes.

We used the HP document for testing.

Our experiments for Financial Covenants showed that:

- BM25 search alone identified 26 out of 31 relevant chunks.
- The KG-enhanced LLM method identified 9 chunks, with 4 overlapping with BM25 results.
- The union of all methods yielded the final set of 31 relevant chunks.

For the General Deal Information:

- BM25 search identified 43 out of 43 relevant chunks.
- The LLM method identified 22 chunks, all overlapping with BM25 results.
- KG relationships provided additional context for more accurate identification.

This hybrid approach leveraged the strengths of all techniques, with BM25 providing broad coverage, the LLM offering more nuanced understanding of complex sections, and the KG adding structured relationships and context.

It's important to note that we don't have a ground truth of the actual relevant chunks. After obtaining these results, we manually checked the highlighted text. We found that it contains some false positives but no false negatives. This provides a good starting point, and we can improve the accuracy moving forward by refining our methods to reduce false positives while maintaining complete coverage of relevant information.

The following JSON block represents a sample of the relationships obtained from the Knowledge Graph that were particularly helpful for identifying financial covenants:

```json
[
  {
    "relationship": "(Node:LegalClause {'Context': 'Total Leverage Ratio and Consolidated EBITDA to Consolidated Net Interest Expense are key financial metrics for Hewlett-Packard.', 'clause': 'Financial Covenants Section', 'Description': 'These covenants outline financial ratios that Hewlett-Packard must maintain, including Total Leverage Ratio and Consolidated EBITDA to Consolidated Net Interest Expense ratio. The Total Leverage Ratio must not exceed 4.00 to 1.0, while the Consolidated EBITDA to Consolidated Net Interest Expense ratio must be at least 3.00 to 1.0.', 'alias': ['Financial Covenant', 'Financial Covenants'], 'definition': 'A clause in Hewlett-Packard\\'s loan agreement that specifies financial conditions.', 'labels': ['Financial Covenants', 'loan covenant']})-[:HAS_DURATION]->(Covenant Period: Node:TemporalConcept {'StartDate': '09/12/2024', 'EndDate': '09/12/2029', 'Duration': '5 years', 'alias': ['Covenant Period'], 'definition': 'The time period during which the financial covenants are applicable.', 'labels': []})"
  },
  {
    "relationship": "(Total Leverage Ratio: Node:FinancialMetric {'MaximumLevel': '4.00 to 1.0', 'alias': ['Total Leverage Ratio'], 'definition': 'A financial covenant ratio measuring Hewlett-Packard\\'s total debt relative to its earnings.', 'labels': []})-[:IS_PART_OF]->(Financial Covenants: Node:LegalClause {'Company': 'Hewlett-Packard', 'StartDate': '09/12/2024', 'EndDate': '09/12/2029', 'Description': 'Financial metrics that Hewlett-Packard must maintain as part of their loan agreement.', 'alias': ['Financial Covenants'], 'labels': ['loan covenant']})"
  },
  {
    "relationship": "(Consolidated EBITDA to Consolidated Net Interest Expense: Node:FinancialMetric {'MinimumLevel': '3.00 to 1.0', 'alias': ['Interest Coverage Ratio'], 'definition': 'A financial covenant ratio measuring Hewlett-Packard\\'s ability to pay interest on outstanding debt.', 'labels': []})-[:HAS_CONSTRAINT]->(Dividend Restriction: Node:FinancialConstraint {'Status': 'N', 'Description': 'Indicates whether there are restrictions on dividend payments based on this financial covenant.', 'alias': ['Dividend Restriction'], 'definition': 'A constraint on dividend payments tied to financial covenant compliance.', 'labels': []})"
  }
]
```

These KG relationships provided valuable context and structure to the LLM prompts, enhancing the identification and understanding of financial covenants within the loan agreement.

## 3. Results and Discussion

Our hybrid approach demonstrated several advantages over alternative methods:

1. **Complexity Handling**: Traditional NLP techniques often struggle with complex entities like "Financial Covenant" in large documents. Our method showed superior performance in identifying nuanced financial terms and concepts, further enhanced by the structured relationships provided by the KG.

2. **Contextual Understanding**: Unlike rule-based systems, our LLM component could grasp the context and semantics of different sections, leading to more accurate identification of relevant text. The KG relationships added another layer of contextual information, improving the overall understanding of the document structure.

3. **Scalability**: The BM25 component allowed for efficient searching across the large document, while the LLM provided depth of understanding where needed. The KG relationships could be easily integrated into the process without significant computational overhead.

4. **Flexibility**: Our approach can be easily adapted to different types of loan agreements and section requirements by adjusting guidelines, keywords, and KG relationships.

Alternative approaches and their limitations:

- **Keyword Matching**: Simple keyword matching would likely result in high false positives due to the complex nature of loan agreement language.
- **Named Entity Recognition (NER)**: While useful for identifying entities, NER alone would struggle with the context-dependent nature of section identification in loan agreements.
- **Topic Modeling**: LDA (Latent Dirichlet Allocation), while useful for topic modeling, lacks the contextual understanding and flexibility needed for comprehensive information extraction from large documents. It's limited to identifying broad topics and word distributions, missing nuanced information and relationships that more advanced models like LLMs can capture and interpret.
- **Document Layout Analysis**: While potentially useful for structured documents, this approach would struggle with the variable formatting and complex language of loan agreements.

Our hybrid method addresses these limitations by combining the broad search capabilities of BM25 with the contextual understanding of an LLM and the structured relationships from a KG, resulting in more accurate and comprehensive section extraction.

To further improve the system, we could implement:

- **Reranking**: Implementing a reranking step after the initial retrieval could help improve precision, especially if queries are formulated properly for each entity type and checked for answer relevance.
- **Query Expansion**: Improving the initial search terms by analyzing top results and incorporating related words or phrases. This could help find more relevant sections that might have been missed initially.
- **Few-shot Learning**: Providing the LLM with a few examples of correctly extracted sections could potentially improve performance without the need for extensive training or adaptation. We already do this but the examples can be improved based on actual user interactions.
- **KG Expansion**: Continuously updating and expanding the Knowledge Graph with new relationships and entities specific to loan agreements could further enhance the contextual understanding and accuracy of the system.

It's important to note that the real accuracy of our approach will be calculated once we have more labeled documents. However, we are confident that our approach looks strong and promising based on the initial results, particularly with the addition of KG relationships enhancing the contextual understanding of the LLM.

## 4. Cost

Our approach utilized the free tier of the Gemini API, which offers the following usage limits:

- 15 RPM (requests per minute)
- 1 million TPM (tokens per minute)
- 1,500 RPD (requests per day)

These limits are more than sufficient for our current needs. To estimate the number of documents that can be processed within these constraints:

Assuming 2 API calls per section and 20 sections to extract per document:

- Total API calls per document: 40
- Documents per minute: 0.375 (15 RPM / 40 calls)
- Documents per hour: 22.5
- Documents per day: 37.5 (limited by RPD: 1,500 / 40 calls)

This demonstrates that even with the free tier, we can process a significant number of loan agreements daily. For increased capacity, upgrading to a paid tier would allow for higher throughput and the ability to handle larger volumes of documents.

For comparison, the paid tier of Gemini API offers significantly higher limits:

- 2,000 RPM (requests per minute)
- 4 million TPM (tokens per minute)
- Prompts up to 128k tokens

Pricing for the paid tier:

- Input: $0.075 / 1 million tokens
- Output: $0.30 / 1 million tokens
- Context Caching: $0.01875 / 1 million tokens

With these higher limits, the potential document processing capacity increases substantially:

- Documents per minute: 50 (2,000 RPM / 40 calls)
- Documents per hour: 3,000
- Documents per day: 72,000

This represents a significant increase in processing capacity compared to the free tier. However, actual usage would depend on specific requirements and budget considerations. The paid tier would be beneficial for organizations needing to process large volumes of loan agreements or requiring faster turnaround times.

It's important to note that these calculations are based on average usage and may vary depending on the complexity of the documents and the specific implementation details of our extraction process. The number of sections per document can significantly impact the processing capacity, and adjustments may be needed for documents with more or fewer sections.

## 5. Conclusion

The proof of concept demonstrated the effectiveness of our hybrid BM25, LLM-based, and KG-enhanced approach for extracting specific sections from large loan agreements. By leveraging the strengths of traditional information retrieval, advanced language models, and structured knowledge representation, we achieved accurate identification and highlighting of relevant text across various complex sections.

This approach shows promise for streamlining the analysis of lengthy legal documents, potentially saving significant time and resources in the loan agreement review process. Future work could focus on further optimizing the balance between BM25, LLM components, and KG integration, expanding the range of identifiable sections, and integrating the system into existing document analysis workflows.