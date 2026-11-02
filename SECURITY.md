# Security

Please report a vulnerability privately, through GitHub's
[private vulnerability reporting](https://github.com/mxlei01/fast-feature-transform/security/advisories/new), rather
than in a public issue. Say which version you used and how to reproduce the problem. Fixes go into the next release;
earlier releases aren't patched.

Reading a Parquet file and loading a saved pipeline both parse input that may come from elsewhere, so a crash, hang
or out-of-bounds access in either is worth reporting. A pickle is different: unpickling can run any code, for this
package's objects as for any other, so only unpickle files you trust. A saved `manifest.json` carries the same fitted
pipeline without that risk.
