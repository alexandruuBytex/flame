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
import sys
from enum import Enum

from pydantic import BaseModel as pydBaseModel
from pydantic import Field

CONF_KEY_CHANNEL_NAME = 'name'
CONF_KEY_CHANNEL_PAIR = 'pair'
CONF_KEY_CHANNEL_IS_BIDIR = 'isBidirectional'
CONF_KEY_CHANNEL_GROUPBY = 'groupBy'
CONF_KEY_CHANNEL_GROUPBY_TYPE = 'type'
CONF_KEY_CHANNEL_GROUPBY_VALUE = 'value'
CONF_KEY_CHANNEL_FUNC_TAGS = 'funcTags'

CONF_KEY_BASE_MODEL = 'baseModel'
CONF_KEY_BASE_MODEL_NAME = 'name'
CONF_KEY_BASE_MODEL_VERSION = 'version'

CONF_KEY_BROKERS = 'brokers'
CONF_KEY_BROKERS_HOST = 'host'
CONF_KEY_BROKERS_SORT = 'sort'

CONF_KEY_JOB = 'job'
CONF_KEY_JOB_ID = 'id'
CONF_KEY_JOB_NAME = 'name'

CONF_KEY_REGISTRY = 'registry'
CONF_KEY_REGISTRY_SORT = 'sort'
CONF_KEY_REGISTRY_URI = 'uri'

CONF_KEY_OPTIMIZER = 'optimizer'
CONF_KEY_OPTIMIZER_SORT = 'sort'
CONF_KEY_OPTIMIZER_KWARGS = 'kwargs'

CONF_KEY_SELECTOR = 'selector'
CONF_KEY_SELECTOR_SORT = 'sort'
CONF_KEY_SELECTOR_KWARGS = 'kwargs'

GROUPBY_DEFAULT_GROUP = 'default'

DEFAULT_HYPERARAMETERS_DICT = {'rounds': 1, 'epochs': 1, 'batchSize': 16}

CONF_KEY_CHANNELCONFIGS = 'channelConfigs'


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

    FEDAVG = 1  # default
    FEDADAGRAD = 2  # FedAdaGrad
    FEDADAM = 3  # FedAdam
    FEDYOGI = 4  # FedYogi
    # FedBuff from https://arxiv.org/pdf/1903.03934.pdf and
    # https://arxiv.org/pdf/2111.04877.pdf
    FEDBUFF = 5
    FEDPROX = 6 # FedProx


class SelectorType(str, Enum):
    """Define selector types."""

    DEFAULT = 1  # default
    RANDOM = 2  # random
    FEDBUFF = 3  # fedbuff


REALM_SEPARATOR = '/'


class Config(object):
    """Config class."""


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
                    if realm.startswith(entry) and (len(realm) == len(entry)
                                                    or realm[len(entry)]
                                                    == REALM_SEPARATOR):
                        return entry

        return GROUPBY_DEFAULT_GROUP


class Broker(FlameSchema):
    sort_to_host: dict


            self.func_tags = dict()
            if CONF_KEY_CHANNEL_FUNC_TAGS in json_data:
                for k, v in json_data[CONF_KEY_CHANNEL_FUNC_TAGS].items():
                    # k: role, v: tag list
                    self.func_tags[k] = v


class ChannelConfigs(FlameSchema):
    backends: dict = Field(default={})
    channel_brokers: dict = Field(default={})


        def __str__(self):
            """Return base model's detail as string."""
            return ("\t--- base model ---\n" +
                    f"\t\t{CONF_KEY_BASE_MODEL_NAME}: {self.name}\n" +
                    f"\t\t{CONF_KEY_BASE_MODEL_VERSION}: {self.version}\n")

    class Brokers(object):
        """Brokers class."""

        def __init__(self, json_data=None):
            """Initialize BaseModel instance."""
            self.sort_to_host = dict()

            for broker in json_data:
                key = broker[CONF_KEY_BROKERS_SORT].upper()
                try:
                    sort = BackendType[key]
                except KeyError:
                    valid_types = [backend.name for backend in BackendType]
                    sys.exit(f"invalid sort type: {key}\n" +
                             f"broker's sort must be one of {valid_types}.")

                host = broker[CONF_KEY_BROKERS_HOST]
                self.sort_to_host[sort] = host

        def __str__(self):
            """Return brokers' details as string."""
            info = ""
            for sort, host in self.sort_to_host.items():
                info += f"\t\t{sort}: {host}\n"
            return ("\t--- brokers ---\n" + info)


        def __init__(self, json_data=None):
            """Initialize Job instance."""
            self.job_id = json_data[CONF_KEY_JOB_ID]
            self.name = json_data[CONF_KEY_JOB_NAME]

        def __str__(self):
            """Return job's detail in string format."""
            return ("\t--- job ---\n" +
                    f"\t\t{CONF_KEY_JOB_ID}: {self.job_id}\n" +
                    f"\t\t{CONF_KEY_JOB_NAME}: {self.name}\n")

    class Registry(object):
        """Registry class."""

        def __init__(self, json_data=None):
            """Initialize Registry instance."""
            sort = json_data[CONF_KEY_REGISTRY_SORT].upper()
            try:
                self.sort = RegistryType[sort]
            except KeyError:
                valid_types = [registry.name for registry in RegistryType]
                sys.exit(f"invailid registry type: {sort}" +
                         f"valid registry type(s) are {valid_types}")

            self.uri = json_data[CONF_KEY_REGISTRY_URI]

        def __str__(self):
            """Return model registry's detail in string format."""
            return ("\t--- registry ---\n" +
                    f"\t\t{CONF_KEY_REGISTRY_SORT}: {self.sort}\n" +
                    f"\t\t{CONF_KEY_REGISTRY_URI}: {self.uri}\n")

    class Selector(object):
        """Selector class."""

        def __init__(self, json_data=None) -> None:
            """Initialize Selector instance."""
            self.sort = SelectorType.DEFAULT
            self.kwargs = dict()

            if CONF_KEY_SELECTOR not in json_data:
                return

            json_data = json_data[CONF_KEY_SELECTOR]

            if CONF_KEY_SELECTOR_SORT not in json_data:
                return

            sort = json_data[CONF_KEY_SELECTOR_SORT].upper()
            try:
                self.sort = SelectorType[sort]
            except KeyError:
                valid_types = [selector.name for selector in SelectorType]
                sys.exit(f"invailid selector type: {sort}" +
                         f"valid selector type(s) are {valid_types}")

            if CONF_KEY_SELECTOR_KWARGS in json_data:
                self.kwargs = json_data[CONF_KEY_SELECTOR_KWARGS]

            if not self.kwargs:
                self.kwargs = dict()

        def __str__(self) -> str:
            """Return model selector's detail in string format."""
            return ("\t--- selector ---\n" +
                    f"\t\t{CONF_KEY_SELECTOR_SORT}: {self.sort}\n" +
                    f"\t\t{CONF_KEY_SELECTOR_KWARGS}: {self.kwargs}\n")

    class Optimizer(object):
        """Optimizer Class."""

        def __init__(self, json_data=None) -> None:
            """Initialize Optimizer instance."""
            self.sort = OptimizerType.FEDAVG
            self.kwargs = dict()


            json_data = json_data[CONF_KEY_OPTIMIZER]

            if CONF_KEY_OPTIMIZER_SORT not in json_data:
                return


            if CONF_KEY_OPTIMIZER_KWARGS in json_data:
                self.kwargs = json_data[CONF_KEY_OPTIMIZER_KWARGS]

        def __str__(self) -> str:
            """Return FL aggregation optimizer's detail in string format."""
            return ("\t--- optimizer ---\n" +
                    f"\t\t{CONF_KEY_OPTIMIZER_SORT}: {self.sort}\n" +
                    f"\t\t{CONF_KEY_OPTIMIZER_KWARGS}: {self.kwargs}\n")


        def __init__(self, json_data=None) -> None:
            """Initialize ChannelConfigs instance."""
            self.backends = dict()
            self.channel_brokers = dict()

            if CONF_KEY_CHANNELCONFIGS not in json_data:
                return
            json_data = json_data[CONF_KEY_CHANNELCONFIGS]

            for k, v in json_data.items():
                backend_key = v['backend'].upper()
                try:
                    self.backends[k] = BackendType[backend_key]
                except:
                    valid_types = [backend.name for backend in BackendType]
                    sys.exit(f"invailid backend type: {backend_key}\n" +
                             f"valid backend type(s) are {valid_types}")
                self.channel_brokers[k] = Config.Brokers(v['brokers'])

    def __init__(self, config_file: str):
        """Initialize Config instance."""
        with open(config_file) as f:
            json_data = json.load(f)
            f.close()

        self.role = json_data[CONF_KEY_ROLE]
        self.realm = json_data[CONF_KEY_REALM]

        self.task = 'local'
        if CONF_KEY_TASK in json_data:
            self.task = json_data[CONF_KEY_TASK]

        self.task_id = json_data[CONF_KEY_TASK_ID]

        self._init_backend(json_data)

        self._init_channels(json_data)

        self._init_hyperparameters(json_data)

        self.brokers = Config.Brokers(json_data[CONF_KEY_BROKERS])

        self.job = Config.Job(json_data[CONF_KEY_JOB])

        self.registry = Config.Registry(json_data[CONF_KEY_REGISTRY])

        self.selector = Config.Selector(json_data)

        self.optimizer = Config.Optimizer(json_data)

        self.channelConfigs = None
        if CONF_KEY_CHANNELCONFIGS in json_data:
            self.channelConfigs = Config.ChannelConfigs(json_data)

        self.dataset = ''
        if CONF_KEY_DATASET in json_data:
            self.dataset = json_data[CONF_KEY_DATASET]

        self.max_run_time = 300
        if CONF_KEY_MAX_RUN_TIME in json_data:
            self.max_run_time = json_data[CONF_KEY_MAX_RUN_TIME]

        self.base_model = None
        if CONF_KEY_BASE_MODEL in json_data:
            self.base_model = Config.BaseModel(json_data[CONF_KEY_BASE_MODEL])

    def _init_backend(self, json_data):
        backend_key = json_data[CONF_KEY_BACKEND].upper()
        try:
            self.backend = BackendType[backend_key]
        except KeyError:
            valid_types = [backend.name for backend in BackendType]
            sys.exit(f"invailid backend type: {backend_key}\n" +
                     f"valid backend type(s) are {valid_types}")

    def _init_channels(self, json_data):
        self.func_tag_map = dict()
        self.channels = dict()

        for channel_info in json_data[CONF_KEY_CHANNEL]:
            channel_config = Config.Channel(channel_info)
            self.channels[channel_config.name] = channel_config

            # build a map from function tag to channel name
            for tag in channel_config.func_tags[self.role]:
                self.func_tag_map[tag] = channel_config.name

    def _init_hyperparameters(self, json_data):
        self.hyperparameters = None
        if CONF_KEY_HYPERPARAMS in json_data:
            self.hyperparameters = json_data[CONF_KEY_HYPERPARAMS]
        for k, v in DEFAULT_HYPERARAMETERS_DICT.items():
            if k in self.hyperparameters:
                continue
            self.hyperparameters[k] = v

    def __str__(self):
        """Return config info as string."""
        info = ("--- config ---\n" +
                f"\t{CONF_KEY_BACKEND}: {self.backend}\n" +
                f"\t{CONF_KEY_TASK}: {self.task}\n" +
                f"\t{CONF_KEY_ROLE}: {self.role}\n" +
                f"\t{CONF_KEY_REALM}: {self.realm}\n" + str(self.base_model) +
                str(self.brokers) + str(self.job) + str(self.registry))
        for _, channel in self.channels.items():
            info += str(channel)

        return info
