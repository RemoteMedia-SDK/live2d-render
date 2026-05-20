# Live2D Cubism Native SDK — one-time CI setup

The `release.yml` workflow needs access to Live2D's
Cubism Native SDK to compile `cubism-core-sys`'s C++ bindings. Live2D's
[Free Material License](https://www.live2d.com/en/sdk/license/)
prohibits redistribution of the SDK, so we can't commit it to this
public repo. Instead it lives in
[`RemoteMedia-SDK/cubism-sdk-internal`](https://github.com/RemoteMedia-SDK/cubism-sdk-internal)
(private) and CI fetches it via a fine-grained PAT.

This is a **one-time** setup per repo. After completing the steps
below, `git push <tag>` triggers a green release build like any other
RemoteMedia-SDK plugin.

## Setup steps

### 1. Generate a fine-grained personal access token

Browser-only because GitHub doesn't expose token creation via API:

1. Open <https://github.com/settings/personal-access-tokens/new>.
2. **Token name**: `cubism-sdk-fetch (live2d-render CI)` — anything
   descriptive; this label appears in audit logs.
3. **Expiration**: pick the longest interval your security policy
   allows (a year is fine). Tokens silently lock CI when they expire;
   put a calendar reminder on rotation day.
4. **Resource owner**: `RemoteMedia-SDK`.
5. **Repository access** → *Only select repositories* →
   `RemoteMedia-SDK/cubism-sdk-internal` (just that one — don't grant
   access to other repos).
6. **Permissions** → *Repository permissions* → set **Contents: Read**.
   Everything else stays *No access*.
7. Click **Generate token**. Copy the resulting `github_pat_…` string;
   you can't view it again after closing the page.

### 2. Add the token as a secret on `live2d-render`

Either via the GitHub UI:

1. <https://github.com/RemoteMedia-SDK/live2d-render/settings/secrets/actions>
2. **New repository secret**
3. **Name**: `CUBISM_SDK_TOKEN`
4. **Secret**: paste the `github_pat_…` value
5. **Add secret**

Or via the `gh` CLI:

```bash
echo "<paste github_pat_… here>" | \
  gh secret set CUBISM_SDK_TOKEN --repo RemoteMedia-SDK/live2d-render
```

### 3. (Optional) verify the build

Trigger the release workflow manually against an existing tag:

```bash
gh workflow run release.yml \
    --repo RemoteMedia-SDK/live2d-render \
    --field tag=v0.1.1
```

Watch:

```bash
gh run watch --repo RemoteMedia-SDK/live2d-render
```

The build should now produce a green `release` run with the
`Fetch Cubism SDK (cache miss)` step succeeding (3–5 s) on the first
matrix-job invocation, and `actions/cache` restoring it on subsequent
runs.

## Rotation

When the PAT expires (or you preemptively rotate):

1. Generate a new PAT with the same scope (step 1 above).
2. Overwrite the secret:

   ```bash
   echo "<new github_pat_… here>" | \
     gh secret set CUBISM_SDK_TOKEN --repo RemoteMedia-SDK/live2d-render
   ```

3. Revoke the old token at <https://github.com/settings/personal-access-tokens>.

## Failure modes

| Symptom in CI log | Diagnosis |
|---|---|
| `CUBISM_SDK_TOKEN secret wasn't passed by the caller workflow` | release.yml's `secrets:` block is missing or misspelled — verify it's at the *job* level (sibling of `with:`), not nested inside `with:`. |
| `gh: HTTP 404: Not Found` from `gh release download` | Either the SDK version tag (`cubism-sdk-version` input) doesn't exist on `cubism-sdk-internal`, or your PAT lacks read access to that repo. |
| `gh: HTTP 401: Bad credentials` | PAT expired. Rotate (see above). |
| Build succeeds but `Live2DRenderNode::new` panics at runtime with "couldn't load libLive2DCubismCore" | The SDK was fetched + extracted but cubism-core-sys's build.rs found the wrong `Core/lib/<platform>/` subdir. Inspect the `Export LIVE2D_CUBISM_CORE_DIR` step's output — should point at the top-level `CubismSdkForNative-X.Y.Z/` dir, not a subdir. |

## Why a private repo and not (alternatives)

- **Vendored in git LFS** — same redistribution problem, just slower clones.
- **S3 bucket with signed URLs** — works but adds external infra; the
  RemoteMedia-SDK org already has GitHub access controls we trust.
- **Self-hosted runner with the SDK pre-installed** — runner reliability
  and concurrency costs more than the per-build cache + fetch.
- **Public stub fallback** — feasible but requires significant refactor
  in `cubism-core-sys` to abstract every C++ binding behind a trait
  with a stub impl. Worth doing if Live2D ever tightens the license
  further, not today.
