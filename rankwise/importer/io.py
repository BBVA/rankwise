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

import importlib
import os
import re

from rankwise.importer.data import InstantiatedObject, UndefinedEnvVarError


def _get_env_var(name):
    try:
        return os.environ[name]
    except KeyError as exc:
        raise UndefinedEnvVarError(
            f"Environment variable {name!r} is not defined", env_var=name
        ) from exc


def _import_from_namespace(namespace, name, globals=None):
    match = re.match(r"(?P<module>[^.]+)\.(?P<class>[^(]+)\((?P<args>.*)\)", name)
    if match:
        try:
            module = importlib.import_module(f"{namespace}.{match.group('module')}")
            class_ = getattr(module, match.group("class"))
        except ImportError as exc:
            raise ImportError(
                f"Could not import module {match.group('module')!r} from namespace {namespace!r}"
            ) from exc
        except AttributeError as exc:
            raise ImportError(
                f"Could not find class {match.group('class')!r} in module {match.group('module')!r}"
            ) from exc
        else:
            eval_globals = (globals if globals else {}) | {
                "class_": class_,
                "ENVVAR": _get_env_var,
            }
            eval_locals = {}
            instance = eval(f"class_({match.group('args')})", eval_globals, eval_locals)
            return InstantiatedObject(expression=name, instance=instance)


def import_embedding_model(expression):
    return _import_from_namespace("llama_index.embeddings", expression)


def import_llm_model(expression):
    return _import_from_namespace("llama_index.llms", expression)


def import_cross_encoder(expression):
    import torch

    return _import_from_namespace("sentence_transformers", expression, globals={"torch": torch})
