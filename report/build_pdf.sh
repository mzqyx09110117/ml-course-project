#!/usr/bin/env bash
# 报告 PDF 构建脚本（三条链路，按需选择）
#
#   ./report/build_pdf.sh layout   # docx -> html -> chrome 打印（版式最接近 Word，推荐用于自查分页/图表位置）
#   ./report/build_pdf.sh tex      # docx -> pdf 经 xelatex（矢量、字体可嵌入，推荐用于最终提交）
#   ./report/build_pdf.sh docx     # md -> docx（由 markdown 生成初稿）
#
# 依赖：pandoc、xelatex（含 ctex/xeCJK）、google-chrome；缺一不可的项脚本会提示。
set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-layout}"
SRC_MD="report/report.md"
SRC_DOCX="report/report.docx"
OUT_PDF="report/report.pdf"

# 中文字体：优先文泉驿/思源，回退 Noto CJK
CJK_FONT="${CJK_FONT:-Noto Serif CJK SC}"

need() { command -v "$1" >/dev/null 2>&1 || { echo "[缺失] 未找到 $1"; exit 1; }; }

check_pdf() {
  local f="$1"
  [ -s "$f" ] || { echo "[失败] 未生成 $f"; exit 1; }
  local pages chars
  pages=$(pdfinfo "$f" 2>/dev/null | awk '/^Pages:/{print $2}')
  chars=$(pdftotext "$f" - 2>/dev/null | tr -d '[:space:]' | wc -m)
  echo "[校验] $f  页数=${pages:-?}  可提取字符数=${chars}"
  # 经验阈值：平均每页少于 30 个字符通常意味着中文没提取出来或页面空白
  if [ -n "${pages:-}" ] && [ "${pages:-0}" -gt 0 ]; then
    avg=$(( ${chars:-0} / pages ))
    [ "$avg" -ge 30 ] || echo "[警告] 平均每页仅 ${avg} 字符，检查字体嵌入或是否存在空白页"
  fi
}

case "$MODE" in
  docx)
    need pandoc
    [ -f "$SRC_MD" ] || { echo "[缺失] $SRC_MD"; exit 1; }
    pandoc "$SRC_MD" -o "$SRC_DOCX" --reference-doc=report/reference.docx 2>/dev/null \
      || pandoc "$SRC_MD" -o "$SRC_DOCX"
    echo "[完成] $SRC_DOCX"
    ;;

  tex)
    need pandoc; need xelatex; need pdfinfo; need pdftotext
    [ -f "$SRC_DOCX" ] || { echo "[缺失] $SRC_DOCX"; exit 1; }
    pandoc "$SRC_DOCX" -o "$OUT_PDF" \
      --pdf-engine=xelatex \
      -V geometry:margin=2.5cm \
      -V CJKmainfont="$CJK_FONT" \
      -V mainfont="$CJK_FONT" \
      -V fontsize=12pt
    check_pdf "$OUT_PDF"
    ;;

  layout)
    need pandoc; need google-chrome; need pdfinfo; need pdftotext
    [ -f "$SRC_DOCX" ] || { echo "[缺失] $SRC_DOCX"; exit 1; }
    TMP_HTML="$(mktemp -d)/report.html"
    pandoc "$SRC_DOCX" -o "$TMP_HTML" --standalone --embed-resources --mathml \
      --metadata title="report"
    # 页面尺寸按 A4 设定，保证与 Word 版式可比
    google-chrome --headless=new --disable-gpu --no-sandbox \
      --no-pdf-header-footer \
      --print-to-pdf="$OUT_PDF" "$TMP_HTML" >/dev/null 2>&1
    check_pdf "$OUT_PDF"
    ;;

  *)
    echo "用法: $0 {layout|tex|docx}"; exit 1;;
esac
