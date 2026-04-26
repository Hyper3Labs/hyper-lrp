# Changelog

## 0.1.0 - 2026-04-26

### Initial Release

- Ship the first standalone `hyper-lrp` package with a working Hugging Face CLIP adapter (`hf-clip`) for image-side AttnLRP explanations.
- Add a small adapter registry, CLI, LXT/Zennit explainer backend, signed heatmap export, and README hero image built from real attribution outputs.
- Pin `transformers` to `4.52.4` for the current LXT AttnLRP integration; the `transformers 5.x` API is not supported by this release.
- Keep HyCoCLIP, MERU, and other model families as future adapters rather than shipping them as implemented targets in this release.

### Notes

- `hyper-lrp` is an early 0.1 package and currently includes one built-in adapter: `hf-clip`.
- LRP methods may be subject to upstream patent and license constraints. Review those constraints before commercial use.