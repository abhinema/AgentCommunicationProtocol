# AgentCommunicationProtocol

## Create venv or conda env
conda create --name acp_env python=3.12

## Installation Instructions
pip install -r requirements.txt

This will install below packages:
Installing collected packages: wcwidth, schema, pytz, pypika, pure-eval, ptyprocess, paginate, mpmath, monotonic, flatbuffers, durationpy, distlib, appdirs, zstandard, zipp, wrapt, websockets, websocket-client, watchdog, uvloop, uv, urllib3, tzdata, typing-extensions, traitlets, tqdm, tomli-w, tomli, tenacity, tabulate, sympy, soupsieve, sniffio, six, shellingham, rpds-py, regex, redis, pyyaml, pytube, python-multipart, python-dotenv, pysbd, pyproject_hooks, pypdfium2, pypdf, pyjwt, pygments, pycparser, pyasn1, pyarrow, psycopg-binary, protobuf, propcache, prompt_toolkit, portalocker, platformdirs, pillow, pexpect, pathspec, parso, packaging, overrides, orjson, opentelemetry-util-http, oauthlib, numpy, nodeenv, networkx, nest-asyncio, mypy-extensions, multidict, mmh3, mkdocs-material-extensions, mergedeep, mdurl, MarkupSafe, markdown, jsonref, jsonpointer, jsonpickle, json5, json-repair, jiter, janus, importlib-resources, idna, identify, hyperframe, humanfriendly, httpx-sse, httptools, hpack, hf-xet, h11, grpcio, fsspec, frozenlist, filelock, fastavro, executing, et-xmlfile, docstring-parser, dnspython, distro, decorator, colorama, click, charset_normalizer, cfgv, certifi, cachetools, blinker, bcrypt, backrefs, backoff, babel, attrs, asttokens, asgiref, annotated-types, aiohappyeyeballs, yarl, virtualenv, uvicorn, typing-inspection, typing-inspect, types-requests, stack_data, sqlalchemy, rsa, requests, referencing, pyyaml-env-tag, python-dateutil, pyright, pymdown-extensions, pydantic-core, pyasn1-modules, psycopg, opentelemetry-proto, openpyxl, obstore, mkdocs-get-deps, matplotlib-inline, marshmallow, markdown-it-py, Mako, load_dotenv, jsonpatch, jinja2, jedi, ipython-pygments-lexers, importlib-metadata, httpcore, h2, googleapis-common-protos, email-validator, deprecation, coloredlogs, chroma-hnswlib, cffi, build, beautifulsoup4, anyio, aiosignal, watchfiles, tiktoken, starlette, rich, requests-toolbelt, requests-oauthlib, pydantic, pre-commit, posthog, pandas, opentelemetry-exporter-otlp-proto-common, opentelemetry-api, onnxruntime, jsonschema-specifications, ipython, huggingface-hub, httpx, gptcache, google-auth, ghp-import, docker, dataclasses-json, cryptography, alembic, aiohttp, typer, tokenizers, smolagents, rich-toolkit, pyvis, pydantic-settings, pdfminer.six, opentelemetry-semantic-conventions, openai, ollama, mkdocs, langsmith, lancedb, kubernetes, jsonschema, fastapi, qdrant-client, pdfplumber, opentelemetry-sdk, opentelemetry-instrumentation, mkdocs-material, litellm, langchain-core, fastapi-cli, cohere, opentelemetry-instrumentation-httpx, opentelemetry-instrumentation-asgi, opentelemetry-exporter-otlp-proto-http, opentelemetry-exporter-otlp-proto-grpc, mem0ai, langchain-text-splitters, langchain-openai, instructor, opentelemetry-instrumentation-fastapi, langchain, langchain-community, chromadb, acp-sdk, langchain-experimental, crewai, langchain-cohere, embedchain, crewai_tools


## RAG Tool
https://docs.crewai.com/en/tools/ai-ml/ragtool

## Ollama
### Server
export OLLAMA_HOST=192.168.0.120:11434
ollama serve

### Client
export OLLAMA_HOST=192.168.0.120:11434
ollama query --host $OLLAMA_HOST "Your query here"

- Server
uv run rag_agent_server.py 
uv run gaurdrail_server.py

- Client
python sequential_client.py 

