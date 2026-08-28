#!/usr/bin/env bash
# Развёртывание публичного демо на чистой Linux-VM (проверялось на Debian 12
# и Ubuntu 24.04 в Google Compute Engine). Подробности и порядок шагов —
# docs/deployment.md.
#
# Два способа доставить код. Из репозитория:
#
#   curl -fsSL https://raw.githubusercontent.com/<аккаунт>/<репозиторий>/main/scripts/deploy_vm.sh | bash -s -- \
#       --repo https://github.com/<аккаунт>/<репозиторий>.git \
#       --host mtbank-demo.duckdns.org \
#       --llm-key gsk_...
#
# Или из уже загруженного на машину каталога (например, распакованного из
# архива через scp) — тогда --repo не нужен, скрипт запускается изнутри:
#
#   cd ~/mtbank-ai-hiring && bash scripts/deploy_vm.sh \
#       --host mtbank-demo.duckdns.org --llm-key gsk_...
#
# Скрипт идемпотентен: повторный запуск обновляет код и пересобирает стек,
# не трогая уже выпущенные сертификаты и накопленную историю звонков.
set -euo pipefail

REPO=""
DEMO_HOSTNAME=""
LLM_API_KEY=""
WHISPER_MODEL="medium"
TARGET_DIR="${HOME}/mtbank-ai-hiring"

usage() {
	cat <<'USAGE'
Использование: deploy_vm.sh --host ИМЯ --llm-key КЛЮЧ [--repo URL] [опции]

  --host ИМЯ          доменное имя демо, на него выпускается сертификат
                      Let's Encrypt (обязательно; голый IP не подойдёт)
  --llm-key КЛЮЧ      ключ OpenAI-совместимого провайдера (обязательно)
  --repo URL          git-репозиторий проекта. Не нужен, если код уже лежит
                      в --dir (загружен через scp) — тогда используется он.
  --whisper-model M   модель faster-whisper, по умолчанию medium
  --dir ПУТЬ          каталог проекта, по умолчанию ~/mtbank-ai-hiring
USAGE
}

while [[ $# -gt 0 ]]; do
	case "$1" in
		--repo) REPO="$2"; shift 2 ;;
		--host) DEMO_HOSTNAME="$2"; shift 2 ;;
		--llm-key) LLM_API_KEY="$2"; shift 2 ;;
		--whisper-model) WHISPER_MODEL="$2"; shift 2 ;;
		--dir) TARGET_DIR="$2"; shift 2 ;;
		-h|--help) usage; exit 0 ;;
		*) echo "Неизвестный аргумент: $1" >&2; usage; exit 1 ;;
	esac
done

for required in DEMO_HOSTNAME LLM_API_KEY; do
	if [[ -z "${!required}" ]]; then
		echo "Не задан обязательный параметр: --${required,,}" >&2
		usage
		exit 1
	fi
done

# Репозиторий не обязателен: код мог приехать архивом через scp. Но если его
# нет ни там, ни там — дальше делать нечего, и сказать об этом надо сразу, а
# не на шаге сборки.
if [[ -z "${REPO}" && ! -f "${TARGET_DIR}/docker-compose.yml" ]]; then
	echo "В ${TARGET_DIR} нет docker-compose.yml, и --repo не задан." >&2
	echo "Либо укажите --repo, либо загрузите код в этот каталог." >&2
	exit 1
fi

# --- 1. Docker -------------------------------------------------------------
# Официальный скрипт вместо пакета из дистрибутива: в репозиториях Debian
# лежит docker.io без плагина compose v2, а нам нужен именно он (в файлах
# используется синтаксис `env_file: [{path, required}]`, требующий 2.24+).
if ! command -v docker >/dev/null 2>&1; then
	echo "==> Ставлю Docker"
	curl -fsSL https://get.docker.com | sh
	sudo usermod -aG docker "$USER" || true
fi

# usermod применится только к новому сеансу, поэтому в этом запуске зовём
# docker через sudo — иначе первый же деплой упрётся в отказ доступа к сокету.
DOCKER="sudo docker"

# --- 2. Код ----------------------------------------------------------------
if [[ -d "${TARGET_DIR}/.git" && -n "${REPO}" ]]; then
	echo "==> Обновляю ${TARGET_DIR}"
	git -C "${TARGET_DIR}" pull --ff-only
elif [[ -n "${REPO}" && ! -d "${TARGET_DIR}" ]]; then
	echo "==> Клонирую в ${TARGET_DIR}"
	git clone "${REPO}" "${TARGET_DIR}"
else
	echo "==> Использую код, уже лежащий в ${TARGET_DIR}"
fi
cd "${TARGET_DIR}"

# --- 3. Конфигурация -------------------------------------------------------
# .env перезаписываем целиком: он выводится из аргументов запуска, и ручные
# правки на сервере должны переживать деплой только через эти аргументы.
echo "==> Пишу .env"
umask 077
cat > .env <<ENVFILE
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_API_KEY=${LLM_API_KEY}
LLM_MODEL=openai/gpt-oss-120b

WHISPER_MODEL=${WHISPER_MODEL}
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=ru

API_HOST=0.0.0.0
API_PORT=8080
MAX_AUDIO_DURATION_SEC=600

DEMO_HOSTNAME=${DEMO_HOSTNAME}
PIPELINES_API_KEY=$(openssl rand -hex 24)
OPENWEBUI_BASE_URL=http://openwebui:8080

LOG_LEVEL=INFO
LOG_JSON=true
OPENWEBUI_UPLOADS_DIR=/openwebui-data/uploads

DIARIZATION_BACKEND=pause_heuristic
WITH_PYANNOTE=false
ENVFILE
umask 022

# --- 4. Запуск -------------------------------------------------------------
echo "==> Собираю и поднимаю стек (первый раз это 10-20 минут)"
${DOCKER} compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

echo
echo "==> Готово. Проверка:"
echo "    чат:     https://${DEMO_HOSTNAME}/"
echo "    API:     https://${DEMO_HOSTNAME}/api/health"
echo "    Swagger: https://${DEMO_HOSTNAME}/api/docs"
echo "    Grafana: https://${DEMO_HOSTNAME}/grafana"
echo
echo "Сертификат Let's Encrypt Caddy выпускает при первом обращении —"
echo "на это уходит до минуты. Логи: sudo docker compose logs -f caddy"
