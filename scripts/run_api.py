import os
import sys
from importlib.util import find_spec
from pathlib import Path

import uvicorn
from dotenv import load_dotenv

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent

load_dotenv(dotenv_path=project_root / ".env")

host = os.getenv("API_HOST", "127.0.0.1")
port = int(os.getenv("API_PORT", "8000"))

log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = (
    '%(levelprefix)s %(asctime)s %(client_addr)s - "%(request_line)s" %(status_code)s'
)

if __name__ == "__main__":
    if find_spec("nlpmed_engine") is None:
        sys.path.append(str(project_root))

    uvicorn.run(
        "nlpmed_engine.api.main:app",
        host=host,
        port=port,
        log_level="info",
        log_config=log_config,
        workers=4,
        reload=False,
    )
