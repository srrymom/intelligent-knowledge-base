# Диаграмма структуры программного продукта

```mermaid
graph TB
    subgraph ext["Внешняя среда"]
        user["👤 Пользователь<br/>(браузер)"]
        ollama["Ollama Server<br/>qwen2.5:3b"]
        hf["Hugging Face Hub<br/>multilingual-e5-small"]
        ffmpeg["FFmpeg"]
        gpu["NVIDIA GPU"]
    end

    subgraph sys["Intelligent Knowledge Base"]

        subgraph gw["Gateway (веб-интерфейс)"]
            app["app.py<br/>Точка входа, сборка UI"]
            handlers["handlers.py<br/>Обработка загрузок и поллинг"]
            fmt["formatting.py<br/>Форматирование транскрипций"]
            mon["monitor.py<br/>Мониторинг ресурсов + Watchdog"]
        end

        subgraph asr_pkg["ASR Worker"]
            asr["worker.py<br/>Распознавание речи (GigaAM)"]
        end

        subgraph llm_pkg["LLM Worker"]
            llm["worker.py<br/>Суммаризация текста"]
        end

        subgraph rag_pkg["RAG Engine"]
            rag["engine.py<br/>Векторный поиск и генерация ответов"]
        end

        subgraph storage_pkg["Storage"]
            kb["kb.py<br/>CRUD базы знаний"]
        end

        subgraph shared_pkg["Shared"]
            cfg["config.py<br/>Конфигурация и пути"]
            log["log.py<br/>Журнал активности"]
        end

        subgraph fs["Файловое хранилище (data/)"]
            q[("queue/")]
            tr[("transcript/")]
            sm[("summary/")]
            kbdir[("knowledge_base/")]
            ragdb[("rag_db/")]
            lock["asr.lock"]
            proc[("processed/")]
            actlog[("activity.log")]
        end
    end

    %% Внешняя среда → система
    user -->|HTTP Gradio| app
    asr -->|CUDA| gpu
    asr -->|декодирование медиа| ffmpeg
    llm -->|REST API суммаризация| ollama
    rag -->|REST API генерация ответа| ollama
    rag -.->|загрузка модели| hf
    mon -->|GET /api/ps| ollama
    mon -->|nvidia-smi| gpu

    %% Внутренние вызовы Gateway
    app --> handlers
    app --> fmt
    app --> mon
    app --> kb
    app --> rag
    app -.->|subprocess + watchdog| asr
    app -.->|subprocess + watchdog| llm
    handlers --> fmt
    handlers -->|индексация| rag

    %% Watchdog: мониторинг и перезапуск воркеров
    mon -->|poll() / restart| asr
    mon -->|poll() / restart| llm

    %% Журнал активности
    asr -->|write_event| log
    llm -->|write_event| log
    log --> actlog
    mon -->|read_tail| actlog

    %% Файловый IPC
    handlers -->|запись медиа| q
    asr -->|чтение и удаление| q
    asr -->|запись транскрипций| tr
    asr -->|создание/удаление| lock
    llm -->|чтение транскрипций| tr
    llm -->|проверка| lock
    llm -->|запись резюме| sm
    handlers -->|чтение и архивирование| sm
    handlers -->|архив транскрипций| proc
    handlers -->|запись записей| kbdir
    kb --> kbdir
    kb -->|удаление из индекса| rag
    rag --> ragdb
    rag -->|переиндексация| kbdir

    %% Стили
    classDef extNode fill:#f9f9f9,stroke:#999,color:#333
    classDef gwNode fill:#d4e6f1,stroke:#2980b9,color:#333
    classDef asrNode fill:#d5f5e3,stroke:#27ae60,color:#333
    classDef llmNode fill:#fef9e7,stroke:#f39c12,color:#333
    classDef ragNode fill:#fadbd8,stroke:#e74c3c,color:#333
    classDef storageNode fill:#e8daef,stroke:#8e44ad,color:#333
    classDef fsNode fill:#fdfefe,stroke:#7f8c8d,color:#333
    classDef sharedNode fill:#d6eaf8,stroke:#1a5276,color:#333

    class user,ollama,hf,ffmpeg,gpu extNode
    class app,handlers,fmt,mon gwNode
    class asr asrNode
    class llm llmNode
    class rag ragNode
    class kb storageNode
    class q,tr,sm,kbdir,ragdb,lock,proc,actlog fsNode
    class cfg,log sharedNode
```
