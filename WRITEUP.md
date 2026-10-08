# Write-up: Verifiable Legal Assistant

## Problem
(2-3 sentences: LLMs invent legal facts/citations; we need every claim traceable.)

## Data
(Which documents, how many, where from. See data/documents.md.)

## Approach
(Pipeline diagram + one paragraph each: chunking, hybrid retrieval, claim-level citations, verifier.)

## Evaluation
- Test set: N questions (answerable + unanswerable), how gold passages were chosen
- Same LLM for every variant: <model>

(Paste eval/results.md here.)

Manual spot-check: we read 20 shown claims by hand; X were truly supported.

## Findings
(Which step helped most? Where does it still fail?)

## Limitations
(e.g. scanned PDFs, English only, verifier uses same LLM, small corpus.)
