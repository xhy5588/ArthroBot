# ArthroBot arm payload calculator: published page

This branch holds the built page that GitHub Pages serves at
https://xhy5588.github.io/ArthroBot/. Its files are generated; do not edit them here.

Source, documentation and build command: `tools/arm_payload/` and `docs/arm_payload.md`
on the `main` branch. To update the page, rebuild on `main`
(`python tools/arm_payload/build_payload_viewer.py`) and copy these files from
`tools/arm_payload/workspace_payload/`: `payload_workspace.html`, `redesign_options.html`,
`any_orientation.csv`, `downward_15deg.csv`, `payload_workspace.png`, `payload_workspace.pdf`.
