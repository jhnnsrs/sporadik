# Releasing sporadik

`sporadik` ships as a PyPI package (`sporadik`). Versioning is automated by
[python-semantic-release][psr] from [Conventional Commits][cc] — you never bump the version by
hand. A push to a release branch runs `.github/workflows/release.yaml`, which:

1. runs the test suite,
2. computes the next version from the commit history, bumps `pyproject.toml`, updates
   `CHANGELOG.md`, tags `vX.Y.Z`, and cuts a GitHub Release,
3. builds the wheel and, **only if a release was cut**, uploads it to PyPI via trusted publishing
   (OIDC).

## Commit messages drive the version

| Commit prefix | Bump | Example |
| --- | --- | --- |
| `fix:` | patch | `fix: refuse an index_order that is not a permutation` |
| `feat:` | minor | `feat: read a slice as an exact byte range` |
| `feat!:` / `BREAKING CHANGE:` footer | **major** | `feat!: rank-general layouts` |

## The spec version is not the package version

`sporadik.SPEC_VERSION` describes the **bytes**; the package version describes the **code**. They
move independently and on purpose:

- A patch or a minor release changes the implementation and leaves every existing store readable.
- A **new spec version** is a change to what a store *is*, so it lands with `SUPPORTED_SPECS`
  gaining a member — never replacing one. A reader that drops a version it used to read makes
  stores unreadable that were valid when written, which is the failure the version exists to
  prevent.
- Bumping the major version does not imply a spec bump, and a spec bump does not imply a major
  version — though in practice one that removes a supported spec would be both.

**A spec change is also a change to other people's implementations.** The format has readers that
cannot import this package — that independence is the point — so the specification in `README.md`
is the artifact they follow, and it has to change in the same commit as `sporadik/spec.py`.
`tests/test_spec_document.py` fails if it does not.

## Branches

`main` cuts stable `X.Y.Z`. `next` cuts prereleases `X.Y.Z-rc.N`, which upload fine and which pip
ignores unless `--pre` is passed. `N.x` maintenance branches cut patch and minor releases for an
older major after a new one has shipped from `main`.

[psr]: https://python-semantic-release.readthedocs.io/
[cc]: https://www.conventionalcommits.org/
