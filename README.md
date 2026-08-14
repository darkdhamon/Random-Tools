# Random Tools

A collection of small, useful tools, scripts, and experiments.

## Repository layout

Each tool should live in its own folder under `tools/`:

```text
tools/
  tool-name/
    README.md
    ...source files
```

Every tool should include a short README explaining:

- what it does;
- its requirements;
- how to install or build it;
- how to run it; and
- any important limitations or safety considerations.

Shared documentation and reusable assets belong in `docs/` and `shared/` respectively. Avoid coupling otherwise independent tools so they remain easy to copy, run, and maintain.

## Adding a tool

1. Create a clearly named directory in `tools/`.
2. Add the source files and a tool-specific `README.md`.
3. Add automated tests when practical.
4. Update the tool index below.

## Tool index

- [OCR PDF](tools/ocr-pdf/README.md) - turn scanned PDFs or ordered image files into searchable PDFs using local OCR.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository conventions.

## License

No license has been selected yet. All rights are reserved unless a license is added later.
