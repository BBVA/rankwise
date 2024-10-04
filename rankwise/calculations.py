# Copyright 2024 Banco Bilbao Vizcaya Argentaria, S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import cache
from hashlib import sha1

from llama_index.core.schema import TextNode


@cache
def content_id(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()


@cache
def content_to_node(text: str) -> TextNode:
    return TextNode(text=text, id_=content_id(text))
