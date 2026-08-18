# Architecture

## First milestone

```text
POST /ai/analyze
      |
      v
Normalizer
  mock now / HyperCLOVA X later
      |
      v
Preservation Validator
  numbers, dates, time, money, URL, email
      |
      v
8-label Diagnosis
  mock now / KcELECTRA after Ground Truth is ready
      |
      v
JSON response
```

## 8-label order

This order must never be changed independently:

1. Task
2. Audience
3. Context
4. Format
5. Tone
6. Length
7. Constraint
8. Example

The canonical definition is `app/core/labels.py`.

## Later phases

Do not add these before the first milestone is stable:

- missing-element priority policy
- HyperCLOVA recommendation generation
- recommendation Validator
- ghost text UX
- accept/edit/reject behavior learning
- final answer generation/validation
- RAG/web fact check
