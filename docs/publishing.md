# Maintaining and publishing the docs

The site is ready to serve from GitHub Pages using `main` → `/docs`. It has been prepared locally; no remote repository, push or publication was performed.

## Edit and rebuild

Markdown is the source of truth. The committed HTML lets GitHub Pages serve the directory directly without a Python or Ruby runtime on the server.

```powershell
python -m pip install -r docs/requirements.txt
python tools/build_docs.py
python tools/build_docs.py --check
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8000`. You can also open `docs/index.html` directly without a server. The existing `simenv` already contains the Markdown dependencies; building the docs does not load MuJoCo, Torch or any model.

Edit the `.md` pages, page list in `docs/site.json`, or stylesheet in `docs/static/site.css`. Rebuild and commit the generated `.html` files alongside the sources. `--check` detects stale or unexpected generated pages, checks internal links and heading anchors, and verifies that the included evidence still matches its recorded hashes. The docs workflow performs this check on pushes and pull requests; it does not deploy.

Paths are relative, so the site works at both a domain root and a project URL such as `https://OWNER.github.io/REPOSITORY/`. There is no custom domain or hard-coded repository URL.

## Enable GitHub Pages later

When the repository has been pushed from a machine with GitHub access:

1. Open the repository's **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**.
3. Select branch **main**, folder **/docs**, then save.
4. Use the site URL shown by GitHub after deployment completes.

`docs/index.html` is the entry point. The empty `docs/.nojekyll` file tells GitHub to serve the prepared static files without Jekyll processing. GitHub serves the committed HTML; it does not run `build_docs.py` in this branch-based setup.

This follows GitHub's official [publishing-source instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site) and [static-site setup](https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site).

## Evidence and local artifacts

Only the selected files listed in [the evidence index](evidence/index.md) are included in the site. Full recordings, observations, source snapshots and checkpoints remain in ignored workspace directories. Archive references marked **local artifact** are provenance paths, not web links.

To add evidence, copy the chosen report/media into `docs/evidence/` or `docs/media/`, record its source path, byte count and SHA-256 in `docs/evidence/manifest.json`, and link it from Markdown. Keep failed and invalid trials labelled, and do not overwrite original experiment records.

## Repository history

The initial commits import the existing simulator/runtime, then the diagnostics and original findings. Subsequent commits reorganize the documentation and add this static site. These are catch-up commits of the current working project; they do not reconstruct or backdate the earlier development process.

Large generated directories (`models/`, `vendor/`, `datasets/`, `runs/`) and local credentials/configuration are ignored. The root milestone specification and selected small scene assets are tracked. Author identity is configured locally for this repository; no global Git configuration was changed.
