# SPDX-FileCopyrightText: Copyright (C) 2025 Omid Jafari <omidjafari.com>
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
workers = int(os.getenv("API_WORKERS", "1"))

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
        workers=workers,
        reload=False,
        timeout_worker_healthcheck=10,
    )
