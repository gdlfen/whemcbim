#!/usr/bin/env bash
# =============================================================================
#  发布到 GitHub —— 同步本地代码到发布副本，然后提交并推送
#
#  用法（在 UCRT64 / Git Bash 终端里）：
#      cd "/e/deepseek desktop/个股预测程序"
#      ./发布.sh                # 正常发布
#      ./发布.sh "改了什么"      # 自定义提交说明
#
#  说明：
#    · 主工作目录是「个股预测程序」，发布副本在「.work/deploy」。
#    · 本脚本只做「主目录 → 发布副本 → GitHub」，不会改动主目录。
#    · 敏感文件（data/ai_config.json 含 API Key）永不推送。
# =============================================================================

set -uo pipefail

# ---------- 路径配置（如有变动只改这里） ----------
SRC="/e/deepseek desktop/个股预测程序"
DEP="/e/deepseek desktop/.work/deploy"
DST="$DEP/个股预测程序"

# ---------- 不推送的内容 ----------
EXCLUDE_DIRS=(.git __pycache__ .pytest_cache .cache .verify_learn _probe .tmp)
EXCLUDE_FILES=(ai_config.json macro_manual.json)

# ---------- 输出辅助 ----------
info() { printf '\033[36m[发布]\033[0m %s\n' "$*"; }
ok()   { printf '\033[32m[成功]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[注意]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[失败]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 前置检查 ----------
[ -d "$SRC" ] || die "找不到主目录：$SRC"
[ -d "$DEP" ] || die "找不到发布副本：$DEP（请先克隆：git clone git@github.com:GDLFEN/WHEMCBIM.git \"$DEP\"）"
cd "$DEP" || die "无法进入 $DEP"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    die "$DEP 不是 git 仓库"
fi

# 远程是否为 SSH（HTTPS 的 443 端口在本机被阻断）
REMOTE_URL="$(git remote get-url origin 2>/dev/null || echo '')"
case "$REMOTE_URL" in
    git@*|ssh://*) : ;;
    https://*)
        warn "远程仍是 HTTPS：$REMOTE_URL"
        warn "本机 443 端口不通，推送会失败。正在自动切换为 SSH…"
        git remote set-url origin git@github.com:GDLFEN/WHEMCBIM.git
        ok "已切换为 git@github.com:GDLFEN/WHEMCBIM.git"
        ;;
    '') die "未配置远程 origin" ;;
esac

# ---------- 1. 同步文件 ----------
info "同步 $SRC  →  $DST"
mkdir -p "$DST" || die "无法创建 $DST"

RSYNC_ARGS=( -a --delete )
for d in "${EXCLUDE_DIRS[@]}";  do RSYNC_ARGS+=( --exclude="$d/" ); done
for f in "${EXCLUDE_FILES[@]}"; do RSYNC_ARGS+=( --exclude="$f" ); done

if command -v rsync >/dev/null 2>&1; then
    rsync "${RSYNC_ARGS[@]}" "$SRC/" "$DST/" || die "rsync 同步失败"
    ok "已用 rsync 同步（含删除多余文件）"
else
    warn "未找到 rsync，退化为 cp（不会删除已移除的文件）"
    # 复制顶层所有条目，跳过排除项
    (
      cd "$SRC" || exit 1
      shopt -s dotglob nullglob
      for item in *; do
          skip=0
          for d in "${EXCLUDE_DIRS[@]}";  do [ "$item" = "$d" ] && skip=1; done
          for f in "${EXCLUDE_FILES[@]}"; do [ "$item" = "$f" ] && skip=1; done
          [ "$skip" = 1 ] && continue
          cp -a "$item" "$DST/" || exit 1
      done
    ) || die "cp 同步失败"
    ok "已用 cp 同步"
fi

# 兜底清理（防止同步带进来）
find "$DST" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null
for f in "${EXCLUDE_FILES[@]}"; do
    for p in "$DST/$f" "$DST/data/$f"; do
        [ -e "$p" ] && rm -f "$p" && warn "已移除敏感/本机文件：${p#$DST/}"
    done
done

# ---------- 2. 提交 ----------
cd "$DEP" || die "无法进入 $DEP"

if [ -z "$(git status --porcelain)" ]; then
    info "没有文件变更，跳过提交"
else
    CHANGED="$(git status --porcelain | wc -l | tr -d ' ')"
    info "检测到 $CHANGED 处变更，准备提交"
    git add -A || die "git add 失败"

    # 最后一道保险：确认没有敏感文件被暂存
    if git diff --cached --name-only | grep -qE 'ai_config\.json|macro_manual\.json'; then
        git reset -q
        die "检测到敏感文件被暂存（ai_config.json / macro_manual.json），已中止"
    fi

    MSG="${1:-deploy: 同步本地代码 $(date '+%Y-%m-%d %H:%M')}"
    git commit -q -m "$MSG" || die "git commit 失败"
    ok "已提交：$MSG"
fi

# ---------- 3. 推送 ----------
info "推送到 GitHub…"
# 用「退出码」判断成功，而不是 grep 文本：
# 管道里用 grep -q 会在匹配后立刻关闭 stdin，令上游 tee 收到 SIGPIPE，
# 配合 pipefail 会误判为失败（这是本脚本初版的 bug）。
PUSH_OUT="$(git push origin main 2>&1)"
PUSH_RC=$?
printf '%s\n' "$PUSH_OUT" | sed 's/^/    /'

if [ "$PUSH_RC" -eq 0 ]; then
    ok "推送完成"
else
    warn "推送失败（退出码 $PUSH_RC），上方为完整输出。"
    cat <<'TIP'

—————————————————————————————————————————————
常见原因与处理：

1) 端口 22 被阻断 → 改用 SSH over 443，执行一次：
     mkdir -p ~/.ssh
     printf 'Host github.com\n  HostName ssh.github.com\n  Port 443\n  User git\n' > ~/.ssh/config
     chmod 600 ~/.ssh/config
     ssh -T git@github.com

2) 密钥未加到 GitHub → 复制公钥后到 https://github.com/settings/keys 添加：
     cat ~/.ssh/id_ed25519.pub

3) 本地拒绝连接到 github.com → 先手工确认：
     ssh -T git@github.com
—————————————————————————————————————————————
TIP
    exit 1
fi

# ---------- 完成 ----------
echo
ok "全部完成"
info "远程仓库：https://github.com/GDLFEN/WHEMCBIM/tree/main/个股预测程序"
git --no-pager log --oneline -1
