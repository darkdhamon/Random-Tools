# Contributing

## Repository conventions

- Keep each tool self-contained in `tools/<tool-name>/`.
- Use lowercase kebab-case for tool directory names.
- Include a README with setup and usage instructions for every tool.
- Do not commit credentials, API keys, personal data, generated build output, or dependency directories.
- Add dependencies only within the tool that needs them.
- Prefer small, focused commits with clear messages.

## Branches and pull requests

Create a branch for each change. Before opening a pull request, run the affected tool's formatter, tests, and build locally. In the pull request, describe what changed and how it was verified.
