# The public website

<https://baconkage.github.io/content-death-clock/>

`index.html` is a static page that reads `data.json` and nothing else. GitHub Pages serves
both from the `gh-pages` branch, which holds only those two files and a `.nojekyll`. It is
kept apart from `main` so the collector's commits every 30 minutes do not rebuild the site.

## Rebuild the numbers

Needs the silver tables and the gold evaluation files (see the main README), then:

```bash
python -m cdc.site.build_data
```

That rewrites `site/data.json`. No video, title or channel is named in it; subscriber counts are
wide bands and views are rounded (`tests/test_site_data.py`).

## Publish a new version

Commit the change on `main`, then copy the two files onto `gh-pages` and push:

```bash
git worktree add ../cdc-gh-pages gh-pages
```

```bash
cp site/index.html site/data.json ../cdc-gh-pages/
```

```bash
git -C ../cdc-gh-pages commit -am "Deploy the public website"
```

```bash
git -C ../cdc-gh-pages push origin gh-pages
```

```bash
git worktree remove ../cdc-gh-pages
```

Pages rebuilds within a minute. To preview locally first, serve the folder rather than opening
the file, because the page fetches `data.json`:

```bash
python -m http.server 8765 --directory site
```
