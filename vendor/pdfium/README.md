# PDFium (prebuilt binary)

This directory holds the prebuilt PDFium library used by `pdfium-render`
for PDF rasterization. None of its contents are checked into git — see
`.gitignore` — only this README. If you are setting up a fresh clone,
download the archive and extract it here.

## Download (Windows x64)

```powershell
cd vendor/pdfium
curl.exe -L -o pdfium-win-x64.tgz `
  "https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz"
tar -xzf pdfium-win-x64.tgz
rm pdfium-win-x64.tgz
```

After extraction, `vendor/pdfium/bin/pdfium.dll` should exist. The Rust
code in `crates/paddleknife-core/src/pdf.rs` locates it automatically
via `find_pdfium_lib_dir()` (fall-back order: `PDFIUM_LIB_PATH` env var,
next to the current exe, then this directory).

## Other platforms

- Linux x64: `pdfium-linux-x64.tgz`
- macOS arm64: `pdfium-mac-arm64.tgz`
- macOS x64: `pdfium-mac-x64.tgz`

Replace the URL in the download command accordingly.

## Upstream

- Binary distribution: https://github.com/bblanchon/pdfium-binaries
- Source: https://pdfium.googlesource.com/pdfium/ (Google Chromium project)

## Licensing

PDFium is licensed under BSD-3-Clause. See `vendor/pdfium/LICENSE` (once
extracted).
