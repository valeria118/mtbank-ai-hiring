# Деплой демо

Основной путь — **обычная VM с `docker compose`** (Google Compute Engine на
пробном периоде, но подойдёт любой Linux-хост с публичным IP). Он выбран не
из экономии: на VM работает ровно тот же `docker compose up`, что описан в
README, со всеми шестью сервисами — включая бонусные Prometheus и Grafana,
которые PaaS-варианты вынуждены выбрасывать.

Ниже по файлу — два альтернативных пути: **Render** (`render.yaml`, если
нужен управляемый PaaS) и **RunPod** (если нужен реальный GPU).

---

## VM + docker compose (основной путь)

### Что понадобится

1. Хост с публичным IP: 4 vCPU / 16 ГБ — комфортно, 2 vCPU / 8 ГБ — минимум
   (`medium` в int8 занимает ~1.5 ГБ, плюс два процесса Whisper — в `api` и
   в `pipelines` — плюс OpenWebUI). На пробном периоде Google Cloud ($300 на
   90 дней) `e2-standard-4` обходится примерно в $3.2 в сутки, то есть
   демо-окно в две недели укладывается в кредиты с большим запасом.
2. **Доменное имя**, указывающее на этот IP. Голый IP не подойдёт:
   Let's Encrypt не выдаёт на него сертификаты, а ТЗ требует HTTPS.
   Бесплатные варианты — [DuckDNS](https://www.duckdns.org)
   (`имя.duckdns.org`, вход через GitHub) или `sslip.io`
   (`34-12-34-56.sslip.io` резолвится в `34.12.34.56` без регистрации
   вообще; минус — общий на всех лимит выпуска сертификатов Let's Encrypt на
   домен `sslip.io`, поэтому DuckDNS надёжнее).
3. Ключ OpenAI-совместимого провайдера (Groq и т.п.).

### 1. Создать VM

```bash
gcloud compute instances create mtbank-demo \
  --machine-type=e2-standard-4 \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-size=50GB \
  --tags=http-server,https-server \
  --zone=europe-west1-b
```

Диск 50 ГБ не с запасом «на всякий случай»: образы `api` и `pipelines` вместе
занимают около 12 ГБ, плюс веса моделей и слои сборки.

Открыть 80 и 443 (теги `http-server`/`https-server` в стандартном проекте уже
привязаны к правилам, но на новом — нет):

```bash
gcloud compute firewall-rules create allow-http-https \
  --allow=tcp:80,tcp:443 --target-tags=http-server,https-server
```

Больше открывать нечего: наружу слушает только Caddy. Все остальные порты
(`8080` API, `3000` чат, `9099` Pipelines, `9090` Prometheus, `3001` Grafana)
в `docker-compose.yml` привязаны к `127.0.0.1` и снаружи не видны.

### 2. Направить имя на IP

Узнать внешний IP: `gcloud compute instances list`. Затем в DuckDNS указать
этот IP для своего поддомена и дождаться, пока `dig +short имя.duckdns.org`
вернёт его. Пока DNS не сошёлся, Caddy не сможет выпустить сертификат.

### 3. Развернуть

Если репозиторий уже создан:

```bash
gcloud compute ssh mtbank-demo --zone=europe-west1-b --command "curl -fsSL https://raw.githubusercontent.com/<аккаунт>/<репозиторий>/main/scripts/deploy_vm.sh | bash -s -- --repo https://github.com/<аккаунт>/<репозиторий>.git --host mtbank-demo.duckdns.org --llm-key gsk_..."
```

Если репозитория ещё нет — залить код архивом и запустить скрипт изнутри
каталога. Исключения обязательны: `.venv`, `build/` и `dist/` весят ~330 МБ и
на сервере бесполезны, а `data/` — локальная история анализов.

```bash
tar --exclude=.venv --exclude=build --exclude=dist --exclude=data \
    --exclude=.git --exclude=__pycache__ --exclude='*.log' \
    -czf /tmp/mtbank.tar.gz -C .. mtbank-ai-hiring
gcloud compute scp /tmp/mtbank.tar.gz mtbank-demo:~ --zone=europe-west1-b
gcloud compute ssh mtbank-demo --zone=europe-west1-b
# уже на машине:
tar -xzf ~/mtbank.tar.gz -C ~
cd ~/mtbank-ai-hiring
bash scripts/deploy_vm.sh --host mtbank-demo.duckdns.org --llm-key gsk_...
```

Скрипт ставит Docker, при необходимости забирает код, пишет `.env` (с
`PIPELINES_API_KEY`, сгенерированным на месте) и поднимает стек командой:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

Первая сборка — 10–20 минут: два тяжёлых образа плюс веса `medium` (~1.5 ГБ,
скачиваются при первом анализе). Прогреть модель заранее, чтобы первый
запрос проверяющего не ждал загрузку:

```bash
curl -s -X POST https://<host>/api/analyze \
  -F "file=@test_data/short_card_block.wav" -o /dev/null -w "%{http_code}\n"
```

Скрипт идемпотентен: повторный запуск обновит код и пересоберёт стек, не
трогая выпущенные сертификаты и накопленную историю звонков.

### 4. Что где

| Адрес | Что это |
| --- | --- |
| `https://<host>/` | чат OpenWebUI — главный сценарий ТЗ |
| `https://<host>/api/analyze` | REST API |
| `https://<host>/api/docs` | Swagger |
| `https://<host>/api/realtime` | демо потоковой транскрибации (бонус) |
| `https://<host>/grafana` | дашборд (бонус), анонимный просмотр |

Порт `9099` (сервер Pipelines) наружу **не** отдаётся сознательно: его
маршруты `GET /{id}/valves` и `POST /{id}/valves/update` объявлены без
аутентификации — см. комментарий в `pipeline.py`. Секретов в валвах нет, но
и подменять через них `LLM_BASE_URL` посторонним незачем.

### 5. Смоук-тест

```bash
curl -s https://<host>/api/health
```

```bash
curl -s -X POST https://<host>/api/analyze -F "file=@test_data/call_dialog_stereo.wav" | head -c 400
```

Затем открыть `https://<host>/`, убедиться, что в списке моделей есть
`MTBank Speech Analytics Pipeline`, приложить аудио в чат и дождаться markdown-отчёта.
Прогнать анализ хотя бы раз до проверки полезно и ради дашборда: Grafana
рисует то, что уже посчитано.

### Известные ограничения этого деплоя

- **Отклик на пятиминутном файле выше 60 секунд.** Локальный прогон на CPU
  дал RTF≈0.42 на `medium` (README, раздел WER), то есть ~126 секунд на
  5 минут моно и вдвое больше на стерео — каждый канал транскрибируется
  отдельно. Требование ТЗ «<60 сек» относится к критерию «Живое демо»
  (5 баллов), а `medium` — к критерию ASR (20 баллов), поэтому выбрана
  модель, а не скорость. Файлы из `test_data/` короче пяти минут и
  укладываются; если отклик критичен — путь один, GPU (раздел RunPod ниже).
- **`WEBUI_AUTH=false`** — вход в чат без регистрации, чтобы проверяющему не
  пришлось заводить аккаунт. На публичном URL это значит, что запросы к
  LLM/ASR может слать любой, кто найдёт ссылку. Для короткого демо-окна
  приемлемо; держать так постоянно — нет.
- **VM нельзя останавливать между проверками.** ТЗ даёт две попытки с
  интервалом 24 часа. Остановленная VM в Google Cloud теряет эфемерный
  внешний IP, а вместе с ним и совпадение с DNS-записью.
- **История звонков живёт на диске VM.** Общая для `api` и `pipelines`
  (оба монтируют `./data`), но при пересоздании VM теряется. Для прод-уровня
  нужна внешняя БД — см. «Известные ограничения» в README.

---

## Render (альтернатива: управляемый PaaS)

`render.yaml` в корне — готовый Blueprint на три сервиса. В отличие от VM,
Render не поднимает `docker-compose.yml` целиком: каждый сервис отдельный, и
Prometheus/Grafana в блюпринт не входят.

### 1. Подключить репозиторий

Render Dashboard → **New → Blueprint** → выбрать репозиторий. Render найдёт
`render.yaml` и предложит создать `mtbank-api`, `mtbank-pipelines`,
`mtbank-openwebui`.

`mtbank-pipelines` объявлен как **приватный сервис** (`type: pserv`): у него
нет публичного адреса, и OpenWebUI ходит к нему по внутренней сети Render на
`http://mtbank-pipelines:9099`. Так закрыт незащищённый эндпоинт валвов.
Оба сервиса, грузящих Whisper, стоят на `plan: standard` — в 512 МБ
`starter`'а `medium` не помещается и контейнер уходит в OOM на загрузке
модели.

### 2. Заполнить секреты

- `LLM_API_KEY` (у `mtbank-api` и `mtbank-pipelines`) — ключ провайдера.
- `OPENAI_API_KEYS` у `mtbank-openwebui` — **то же значение**, что Render
  сгенерировал для `PIPELINES_API_KEY` у `mtbank-pipelines`
  (`generateValue: true`; посмотреть можно в Dashboard → mtbank-pipelines →
  Environment после первого деплоя).
- `OPENWEBUI_API_KEY` у `mtbank-pipelines` — **обязателен именно на Render**,
  иначе загрузка аудио в чат упадёт с 401. Локально вложения читаются с
  общего тома `openwebui_data`, но на Render общего диска между сервисами
  нет, и единственный путь к файлу — HTTP `/api/v1/files/{id}/content`,
  который требует авторизацию даже при `WEBUI_AUTH=false` (проверено на
  OpenWebUI 0.11). Ключ выпускается уже после первого деплоя:
  `https://mtbank-openwebui.onrender.com` → Settings → Account → API keys,
  затем вписать в Environment у `mtbank-pipelines` и передеплоить.

  > Порядок вынужденный: ключ невозможно выпустить до того, как OpenWebUI
  > поднимется. То есть первый деплой чата заведомо неполный — REST API
  > работает сразу, а чат принимает файлы только после этого шага.

### 3. Проверить URL

Render называет публичный адрес по полю `name`. Если имя занято и Render
присвоил другой поддомен — обновить `OPENWEBUI_BASE_URL` (у `mtbank-api` и
`mtbank-pipelines`) вручную и передеплоить. Адрес приватного сервиса
(`http://mtbank-pipelines:9099` в `OPENAI_API_BASE_URLS`) от поддомена не
зависит и правки не требует.

### 4. Смоук-тест

```bash
curl -s https://mtbank-api.onrender.com/health
```

Открыть `https://mtbank-openwebui.onrender.com`, проверить наличие модели
`MTBank Speech Analytics Pipeline` и загрузить аудио.

### Ограничения именно Render-деплоя

- **SQLite не расшарен между `mtbank-api` и `mtbank-pipelines`** — это два
  сервиса с независимыми дисками. `GET /trends` и команда «тренды» в чате
  увидят только звонки, обработанные тем сервисом, куда пришёл запрос.
  Диск (`disk:`) у каждого свой и переживает деплой, но общим не становится.
- **Prometheus/Grafana не задеплоены** — шаблон в комментарии внизу
  `render.yaml`.
- **Холодный старт.** Неактивный сервис засыпает, и первый запрос после
  паузы может занять существенно дольше обычного. Перед проверкой стоит
  прогреть сервис коротким запросом.

---

## RunPod (альтернатива: если нужен GPU)

RunPod даёт HTTPS-прокси на каждый открытый порт
(`https://{pod-id}-{port}.proxy.runpod.net`) и реальный GPU — в этом варианте
`faster-whisper` использует батчинг (`BatchedInferencePipeline`) и
`WHISPER_MODEL=medium` укладывается в «<60 сек» с запасом, чего на CPU не
добиться.

### 1. Создать под

- Тип: **GPU Pod**, Community Cloud.
- GPU: RTX 3090 / RTX 4090 (`medium` требует ~2 ГБ VRAM в `float16` —
  подойдёт почти любая карта, берём по цене).
- Template: **RunPod Pytorch** (CUDA-драйверы и Docker уже внутри).
- Disk: 30 ГБ (веса `medium` ~1.5 ГБ + образы).
- Exposed HTTP ports: `3000` (OpenWebUI), `8080` (REST API), `3001` (Grafana).

### 2. Развернуть

```bash
git clone https://github.com/<аккаунт>/<репозиторий>.git
cd mtbank-ai-hiring
cp .env.example .env
nano .env    # LLM_API_KEY, WHISPER_DEVICE=cuda, WHISPER_COMPUTE_TYPE=float16
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

`docker-compose.gpu.yml` — оверлей с GPU-резервацией
(`deploy.resources.reservations.devices`). Он **не** слит в основной файл
намеренно: обычный `docker compose up` должен подниматься без GPU, а жёсткая
резервация уронила бы запуск на машине без `nvidia-container-toolkit`.

Первый запуск — 5–10 минут.

### 3. Проверить

```bash
curl -s https://<pod-id>-8080.proxy.runpod.net/health
```

Открыть `https://<pod-id>-3000.proxy.runpod.net` — в списке моделей должна
быть `MTBank Speech Analytics Pipeline` (идентификатор пайплайна — `mtbank_pipeline`).

### 4. Прогреть модель

```bash
curl -s -X POST https://<pod-id>-8080.proxy.runpod.net/analyze \
  -F "file=@test_data/short_card_block.wav" -o /dev/null -w "%{http_code}\n"
```

### 5. Не останавливать под

Доступность демо проверяют до двух раз с интервалом 24 часа, а при остановке
RunPod меняет URL прокси.

### 6. Локальная проверка GPU-режима (опционально)

Если есть локальная NVIDIA GPU с `nvidia-container-toolkit`:

```bash
WHISPER_DEVICE=cuda WHISPER_COMPUTE_TYPE=float16 \
  docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build api
docker compose logs api --tail 30 | grep -i "model.load"
```

Ожидаемо: `model.load.done` с `device: cuda`. При нехватке VRAM — понизить
`WHISPER_COMPUTE_TYPE` до `int8_float16`.
