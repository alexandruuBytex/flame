# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Config parser."""

import json
import typing as t
from enum import Enum
from pydantic import Field
import typing as t
from pydantic import BaseModel as pydBaseModel
import json
import typing as t
from enum import Enum

from pydantic import BaseModel as pydBaseModel
from pydantic import Field

from pydantic import BaseModel as pydBaseModel
from pydantic import Field


class FlameSchema(pydBaseModel):
    pass


GROUPBY_DEFAULT_GROUP = "default"
REALM_SEPARATOR = "/"
DEFAULT_HYPERARAMETERS_DICT = {"rounds": 1, "epochs": 1, "batchSize": 16}


class BackendType(str, Enum):
    """Define backend types."""

    LOCAL = "local"
    P2P = "p2p"
    MQTT = "mqtt"


class RegistryType(str, Enum):
    """Define model registry types."""

    DUMMY = "dummy"
    MLFLOW = "mlflow"


class OptimizerType(str, Enum):
    """Define optimizer types."""

    FEDAVG = "fedavg"
    FEDADAGRAD = "fedadagrad"
    FEDADAM = "fedadam"
    FEDYOGI = "fedyogi"
<<<<<<< HEAD
    # FedBuff from https://arxiv.org/pdf/1903.03934.pdf and
    # https://arxiv.org/pdf/2111.04877.pdf
    FEDBUFF = "fedbuff"
    FEDPROX = "fedprox"  # FedProx
=======
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c

    DEFAULT = FEDAVG


class SelectorType(str, Enum):
    """Define selector types."""

    DEFAULT = "default"
    RANDOM = "random"
<<<<<<< HEAD
    FEDBUFF = "fedbuff"
=======
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c


class Job(FlameSchema):
    job_id: str = Field(alias="id")
    name: str


class Registry(FlameSchema):
    sort: RegistryType
    uri: str


class Selector(FlameSchema):
    sort: SelectorType = Field(default=SelectorType.DEFAULT)
    kwargs: dict = Field(default={})


class Optimizer(FlameSchema):
    sort: OptimizerType = Field(default=OptimizerType.DEFAULT)
    kwargs: dict = Field(default={})


class BaseModel(FlameSchema):
<<<<<<< HEAD
    name: str = Field(default="")
    version: int = Field(default=0)
=======
    name: str
    version: int
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c


class Hyperparameters(FlameSchema):
    batch_size: t.Optional[int] = Field(alias="batchSize")
    learning_rate: t.Optional[float] = Field(alias="learningRate")
    rounds: int
    epochs: int
<<<<<<< HEAD
    aggregation_goal: t.Optional[int] = Field(alias="aggGoal", default=None)
=======
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c


class Groups(FlameSchema):
    param_channel: str
    global_channel: str


class FuncTags(FlameSchema):
    aggregator: list[str]
    trainer: list[str]


class GroupBy(FlameSchema):
    type: t.Optional[str] = Field(default="")
    value: t.Optional[list[str]] = Field(default=[])

    def groupable_value(self, realm=""):
        """Return groupby value."""
        if self.value is None:
            return GROUPBY_DEFAULT_GROUP

        for entry in self.value:
            # check if an entry is a prefix of realm in a '/'-separated
            # fashion; if so, then return the matching entry
<<<<<<< HEAD
            if realm.startswith(entry) and (len(realm) == len(entry)
                                            or realm[len(entry)]
                                            == REALM_SEPARATOR):
=======
            if realm.startswith(entry) and (
                len(realm) == len(entry) or realm[len(entry)] == REALM_SEPARATOR
            ):
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c
                return entry

        return GROUPBY_DEFAULT_GROUP


class Broker(FlameSchema):
    sort_to_host: dict


class Channel(FlameSchema):
    name: str
    pair: list[str] = Field(min_length=2)
    is_bidirectional: t.Optional[bool] = Field(default=True)
    group_by: t.Optional[GroupBy] = Field(default=GroupBy())
<<<<<<< HEAD
    func_tags: dict = Field(default={}, alias="func_tags")
=======
    func_tags: dict = Field(default={}, alias="funcTags")
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c
    description: t.Optional[str]


class ChannelConfigs(FlameSchema):
    backends: dict = Field(default={})
    channel_brokers: dict = Field(default={})


<<<<<<< HEAD
class Model(FlameSchema):
    base_model: BaseModel
    optimizer: t.Optional[Optimizer] = Field(default=Optimizer())
    selector: Selector
    hyperparameters: Hyperparameters
    dependencies: list[str]


=======
<<<<<<< HEAD
>>>>>>> 97c8fd08ddb0df794e637bd5a8cd7ca2648b34e0
class Config(FlameSchema):

    def __init__(self, config_path: str):
        raw_config = read_config(config_path)
        transformed_config = transform_config(raw_config)

        super().__init__(**transformed_config)

=======
class Model(FlameSchema):
    base_model: BaseModel
    optimizer: t.Optional[Optimizer] = Field(default=Optimizer())
    selector: Selector
    hyperparameters: Hyperparameters
    dependencies: list[str]


class Config(FlameSchema):
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c
    role: str
    realm: str
    task: t.Optional[str] = Field(default="local")
    task_id: str
    backend: BackendType
    channels: dict
<<<<<<< HEAD
=======
<<<<<<< HEAD
    hyperparameters: Hyperparameters
>>>>>>> 97c8fd08ddb0df794e637bd5a8cd7ca2648b34e0
    brokers: Broker
    job: Job
    model: Model
    registry: Registry
    channel_configs: t.Optional[ChannelConfigs]
    dataset: str
    max_run_time: int
    groups: t.Optional[Groups]
<<<<<<< HEAD
=======
    dependencies: list[str]
=======
    brokers: Broker
    job: Job
    model: Model
    registry: Registry
    channel_configs: t.Optional[ChannelConfigs]
    dataset: str
    max_run_time: int
    groups: t.Optional[Groups]
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c
>>>>>>> 97c8fd08ddb0df794e637bd5a8cd7ca2648b34e0
    func_tag_map: t.Optional[dict]


def read_config(filename: str) -> dict:
    with open(filename) as f:
        return json.loads(f.read())


<<<<<<< HEAD
def transform_config(raw_config: dict) -> dict:
=======
def load_config(filename: str) -> Config:
    raw_config = read_config(filename)
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c
    config_data = {
        "role": raw_config["role"],
        "realm": raw_config["realm"],
        "task_id": raw_config["taskid"],
        "backend": raw_config["backend"],
    }

    if raw_config.get("task", None):
        config_data = config_data | {
            "task": raw_config["task"],
        }

<<<<<<< HEAD
    channels, func_tag_map = transform_channels(config_data["role"],
                                                raw_config["channels"])
    config_data = config_data | {
        "channels": channels,
        "func_tag_map": func_tag_map
    }

    hyperparameters = transform_hyperparameters(raw_config["hyperparameters"])
    config_data = config_data | {"hyperparameters": hyperparameters}

=======
    channels, func_tag_map = transform_channels(
        config_data["role"], raw_config["channels"]
    )
    config_data = config_data | {
        "channels": channels,
        "func_tag_map": func_tag_map,
    }

>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c
    sort_to_host = transform_brokers(raw_config["brokers"])
    config_data = config_data | {"brokers": sort_to_host}

    config_data = config_data | {
        "job": raw_config["job"],
        "registry": raw_config["registry"],
<<<<<<< HEAD
=======
<<<<<<< HEAD
        "selector": raw_config["selector"],
>>>>>>> 97c8fd08ddb0df794e637bd5a8cd7ca2648b34e0
    }

    backends, channel_brokers = transform_channel_configs(
        raw_config.get("channelConfigs", {}))
    config_data = config_data | {
        "channel_configs": {
            "backends": backends,
            "channel_brokers": channel_brokers
=======
    }

    backends, channel_brokers = transform_channel_configs(
        raw_config.get("channelConfigs", {})
    )
    config_data = config_data | {
        "channel_configs": {
            "backends": backends,
            "channel_brokers": channel_brokers,
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c
        }
    }

    config_data = config_data | {
        "dataset": raw_config.get("dataset", ""),
        "max_run_time": raw_config.get("maxRunTime", 300),
<<<<<<< HEAD
=======
<<<<<<< HEAD
        "base_model": raw_config.get("baseModel", None),
        "dependencies": raw_config.get("dependencies", None),
>>>>>>> 97c8fd08ddb0df794e637bd5a8cd7ca2648b34e0
    }

    return config_data
=======
    }

    config_data = config_data | {
        "model": transform_model(raw_config["modelSpec"])
    }

    return Config(**config_data)


def transform_model(raw_model_config: dict):
    base_model = raw_model_config["baseModel"]
    optimizer = raw_model_config.get("optimizer", {})
    selector = raw_model_config["selector"]
    hyperparameters = transform_hyperparameters(
        raw_model_config["hyperparameters"]
    )
    dependencies = raw_model_config.get("dependencies", [])

    return {
        "base_model": base_model,
        "optimizer": optimizer,
        "selector": selector,
        "hyperparameters": hyperparameters,
        "dependencies": dependencies,
    }
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c


def transform_model(raw_model_config: dict):
    base_model = raw_model_config["baseModel"]
    optimizer = raw_model_config.get("optimizer", {})
    selector = raw_model_config["selector"]
    hyperparameters = transform_hyperparameters(
        raw_model_config["hyperparameters"]
    )
    dependencies = raw_model_config.get("dependencies", [])

    return {
        "base_model": base_model,
        "optimizer": optimizer,
        "selector": selector,
        "hyperparameters": hyperparameters,
        "dependencies": dependencies,
    }


def transform_channel(raw_channel_config: dict):
    name = raw_channel_config["name"]
    pair = raw_channel_config["pair"]
    is_bidirectional = raw_channel_config.get("isBidirectional", True)
<<<<<<< HEAD
    group_by = {
        "type": "",
        "value": []
    } | raw_channel_config.get("groupBy", {})
=======
    group_by = {"type": "", "value": []} | raw_channel_config.get("groupBy", {})
>>>>>>> d161660e15d0be038af15bca301ef9e41e023a3c
    func_tags = raw_channel_config.get("funcTags", {})
    description = raw_channel_config.get("description", "")

    return {
        "name": name,
        "pair": pair,
        "is_bidirectional": is_bidirectional,
        "group_by": group_by,
        "func_tags": func_tags,
        "description": description,
    }


def transform_channels(role, raw_channels_config: dict):
    channels = {}
    func_tag_map = {}
    for raw_channel_config in raw_channels_config:
        channel = transform_channel(raw_channel_config)
        channels[channel["name"]] = Channel(**channel)

        for tag in channel["func_tags"][role]:
            func_tag_map[tag] = channel["name"]

    return channels, func_tag_map


def transform_hyperparameters(raw_hyperparameters_config: dict):
    hyperparameters = DEFAULT_HYPERARAMETERS_DICT
    if raw_hyperparameters_config:
        hyperparameters = hyperparameters | raw_hyperparameters_config

    return hyperparameters


def transform_brokers(raw_brokers_config: dict):
    sort_to_host = {}
    for raw_broker in raw_brokers_config:
        sort = raw_broker["sort"]
        host = raw_broker["host"]
        sort_to_host[sort] = host

    return Broker(sort_to_host=sort_to_host)


def transform_channel_configs(raw_channel_configs_config: dict):
    backends = {}
    channel_brokers = {}

    for k, v in raw_channel_configs_config.items():
        backends[k] = v["backend"]
        channel_brokers[k] = transform_brokers(v["brokers"])

    return backends, channel_brokers
