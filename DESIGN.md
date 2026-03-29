# The Pile

Document organizer for complex documents.

## Functions

* Personal document storage (IDs, contracts, agreenets, etc.)
* Structural extraction (personal details, income and expenses, etc.)
* (Maybe) integration with other services (e.g. bank accounts)
* Search over structured and unstructured information
* (Maybe) automatic form filling

Typical use cases include filling of the tax forms, applications, contracts, etc.

All intelligent features are powered by LLMs.

## Components

* Document storage - place or reference to stored informations (source PDFs, images, etc.)
* Extractor - LLM-based mapper from source documents to a tree of structured data
* Search - search over structured and unstructured information with grounding (document -> page -> bbox)