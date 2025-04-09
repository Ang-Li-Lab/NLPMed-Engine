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

"""Main entry point for the NLPMed API.

This module initializes the FastAPI application for the NLPMed API, sets up CORS
middleware, and includes API routes for medical text processing. It also provides
a health check endpoint to verify that the API is running.

Modules:
    FastAPI: The main framework used to create the API.
    CORSMiddleware: Middleware for handling Cross-Origin Resource Sharing (CORS).
    router: The router module that contains all API routes for NLPMed-Engine.

Attributes:
    app (FastAPI): The main FastAPI application instance.

Middleware:
    CORSMiddleware: Configured to allow requests from any origin, restricted to GET
    and POST methods without credentials.

Routes:
    router: Includes routes defined in the `nlpmed_engine.api.routes`.

Endpoints:
    / (GET): Health check endpoint that returns the status of the API.

Usage:
    Run this module to start the NLPMed API server.

"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nlpmed_engine.api.routes import router

app = FastAPI(title="NLPMed API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Health Check"])
def health_check() -> dict:
    """Health check endpoint for the NLPMed API.

    This endpoint provides a simple health check to verify that the API is running and
    accessible. It returns a status message indicating the operational state of the API.

    Returns:
        dict: A JSON response containing the status of the API, typically indicating
        that the API is running.

    Example Response:
        {
            "status": "API is running"
        }

    Tags:
        Health Check

    """
    return {"status": "API is running"}
