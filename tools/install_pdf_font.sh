#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
target="$PWD/eval_platform/assets/fonts"
mkdir -p "$target"
if [ -f "$target/wqy-microhei.ttc" ]; then
  echo "PDF Chinese font already available"
  exit 0
fi
work="$PWD/.font-package"
mkdir -p "$work"
cd "$work"
apt-get download fonts-wqy-microhei
package=$(find "$work" -maxdepth 1 -type f -name 'fonts-wqy-microhei_*.deb' -print -quit)
test -n "$package"
dpkg-deb -x "$package" extracted
cp extracted/usr/share/fonts/truetype/wqy/wqy-microhei.ttc "$target/"
cp extracted/usr/share/doc/fonts-wqy-microhei/copyright "$target/COPYRIGHT.wqy-microhei"
echo "PDF font installed in $target (no system packages changed)"
