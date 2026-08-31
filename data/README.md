# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

The intended distribution is `catalog.jsonl.gz` plus `SHA256SUMS` in a GitHub
Release. No release was published as of 2026-08-29. When an organizer-authorized
archive is available, verify its checksum and decompress it as `catalog.jsonl`
in this directory. Expected row count: 50,000. See the root README and
`docs/reproduction.md` for validation commands.

Never place API keys, private evaluation data, or participant outputs in this directory.
