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

os.environ["API_ML_MODEL_NAMES"] = "TEST"
os.environ["API_ML_TEST_DEVICE"] = "cpu"
os.environ["API_ML_TEST_MODEL_PATH"] = "prajjwal1/bert-mini"
os.environ["API_ML_TEST_TOKENIZER_PATH"] = "prajjwal1/bert-mini"
os.environ["API_ML_TEST_MAX_LENGTH"] = "128"
