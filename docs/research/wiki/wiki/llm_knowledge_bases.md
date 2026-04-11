# LLM Knowledge Bases

LLM Knowledge Bases are personal or research-focused knowledge bases built and maintained using Large Language Models (LLMs). This approach leverages LLMs to organize, compile, and query extensive collections of knowledge efficiently.

## Overview

This concept involves indexing various source documents such as articles, papers, repositories, datasets, and images into a raw data directory. An LLM is then used to incrementally compile a wiki, which consists of markdown (.md) files organized in a directory structure. The wiki includes:

- Summaries of all raw data
- Backlinks between related topics
- Categorization of data into concepts
- Articles written for each concept
- Cross-links among articles

## Data Ingestion

Source documents are collected and indexed into a raw/ directory. Tools like the Obsidian Web Clipper extension can convert web articles into markdown files, with associated images downloaded locally for easy reference by the LLM.

## IDE and Viewing

Obsidian is used as the integrated development environment (IDE) frontend, allowing users to view raw data, the compiled wiki, and derived visualizations. The LLM writes and maintains the data in the wiki, with minimal direct manual edits.

## Question & Answering (Q&A)

Once the wiki grows sufficiently large (e.g., 100 articles and around 400,000 words), the LLM can answer complex questions by researching the compiled knowledge base. This process can often be done without advanced Retrieval-Augmented Generation (RAG) techniques, as the LLM maintains indexes and summaries effectively.

## Output Formats

Outputs from queries can be rendered in various formats such as:

- Markdown files
- Slide shows (e.g., using Marp format)
- Visualizations (e.g., matplotlib images)

These outputs are viewable in Obsidian and can be "filed" back into the wiki to enrich it further.

## Linting and Maintenance

LLMs can perform health checks on the wiki to:

- Identify inconsistent or missing data
- Impute missing information (possibly using web searches)
- Discover interesting connections for new article creation

This incremental cleanup enhances the data integrity and usefulness of the knowledge base.

## Additional Tools

Additional tools can be developed for data processing, such as simple search engines over the wiki. These tools can be used directly or handed off to the LLM for more complex queries.

## Future Directions

As the knowledge base grows, there is potential for synthetic data generation and fine-tuning of LLMs to embed knowledge directly into the model weights, reducing reliance on context windows.

## Summary

In summary, LLM Knowledge Bases involve collecting raw data, compiling it into a markdown wiki using an LLM, and operating on it through various command-line interfaces and queries. The entire system is integrated into tools like Obsidian, with the LLM primarily responsible for writing and maintaining the wiki content. This approach holds promise for developing powerful new products beyond simple script collections.

---

*Source: [Andrej Karpathy's tweet](https://x.com/karpathy/status/2039805659525644595)*
*Author: Andrej Karpathy (@karpathy)*
*Date: Thu Apr 02 20:42:21 +0000 2026*