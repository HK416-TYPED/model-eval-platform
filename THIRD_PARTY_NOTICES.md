# Third-party components

## Bundled font

`eval_platform/assets/fonts/wqy-microhei.ttc` is WenQuanYi Micro Hei 0.2.0 beta.
The full copyright and licensing terms, including upstream contributions and
the GPL font exception / Apache license terms, are preserved in
[`COPYRIGHT.wqy-microhei`](eval_platform/assets/fonts/COPYRIGHT.wqy-microhei).

## External model runtimes

Model runtimes, model weights and datasets are not included in this source archive.
Obtain them from their authors and retain their respective licenses and access terms.

- Anima V7 training project: <https://github.com/Yidhar/anima-native-context-v7>
- Official Krea2 runtime: <https://github.com/krea-ai/krea-2>
- Krea2 sampler contract test snapshot: `db3984fbc6e13b34c0064990fc2d95ac64d00058`

The Anima adapter uses the external native-context V7 runtime API. It validates
the checkpoint contract and preserves numerical sampling functions. Its small
descriptor bridge is documented in `eval_platform/backends.py`.

Python package dependencies are declared in `pyproject.toml`; each retains its
own license. No license for this project's original source has been selected yet.
