"""Module for testing the validation module for the dataset config"""

import warnings
from typing import Optional

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.datasets import ChatTemplate

warnings.filterwarnings("error")


@pytest.fixture(name="minimal_cfg")
def fixture_cfg():
    return DictDefault(
        {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            "learning_rate": 0.000001,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
        }
    )


# pylint: disable=too-many-public-methods (duplicate-code)
class BaseValidation:
    """
    Base validation module to setup the log capture
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def _check_dataset(self, checked_cfg, cfg, dataset_index: int = 0, expect_global_chat_template=None, expect_dataset_chat_template=None):
        """Helper to assert common dataset/config expectations used across tests.

        - If expect_global_chat_template is False, asserts checked_cfg.chat_template is None.
        - If expect_global_chat_template is not None and not False, asserts equality with checked_cfg.chat_template.
        - If expect_dataset_chat_template is not None, asserts equality with dataset.chat_template.
        - Asserts common fields (path, type) and optional fields if present.
        """
        ds_checked = checked_cfg.datasets[dataset_index]
        ds_cfg = cfg.datasets[dataset_index]

        assert ds_checked.path == ds_cfg.path
        assert ds_checked.type == ds_cfg.type

        if hasattr(ds_cfg, "shards"):
            assert ds_checked.shards == ds_cfg.shards

        # Global chat template expectations
        if expect_global_chat_template is False:
            assert checked_cfg.chat_template is None
        elif expect_global_chat_template is not None:
            assert checked_cfg.chat_template == expect_global_chat_template

        # Dataset-level chat template expectations
        if expect_dataset_chat_template is not None:
            assert ds_checked.chat_template == expect_dataset_chat_template

        # Optional per-dataset fields
        if hasattr(ds_cfg, "field_messages"):
            assert ds_checked.field_messages == ds_cfg.field_messages
        if hasattr(ds_cfg, "message_field_role"):
            assert ds_checked.message_field_role == ds_cfg.message_field_role
        if hasattr(ds_cfg, "message_field_content"):
            assert ds_checked.message_field_content == ds_cfg.message_field_content


class TestValidationCheckDatasetConfig(BaseValidation):
    """
    Test the validation for the dataset config to ensure no correct parameters are dropped
    """

    def test_dataset_config_no_drop_param(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                        "shards": 10,
                    }
                ]
            }
        )

        checked_cfg = validate_config(cfg)

        self._check_dataset(checked_cfg, cfg)

        checked_cfg = validate_config(
            cfg,
            capabilities={
                "bf16": "false",
                "n_gpu": 1,
                "compute_capability": "8.0",
            },
            env_capabilities={
                "torch_version": "2.5.1",
            },
        )

        self._check_dataset(checked_cfg, cfg)

    def test_dataset_default_chat_template_no_drop_param(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "shards": 10,
                        "message_field_role": "from",
                        "message_field_content": "value",
                    }
                ],
            }
        )

        checked_cfg = validate_config(cfg)

        self._check_dataset(
            checked_cfg,
            cfg,
            expect_global_chat_template=False,
            expect_dataset_chat_template=ChatTemplate.tokenizer_default,
        )

        checked_cfg = validate_config(
            cfg,
            capabilities={
                "bf16": "false",
                "n_gpu": 1,
                "compute_capability": "8.0",
            },
            env_capabilities={
                "torch_version": "2.5.1",
            },
        )

        self._check_dataset(
            checked_cfg,
            cfg,
            expect_global_chat_template=False,
            expect_dataset_chat_template=ChatTemplate.tokenizer_default,
        )

    def test_dataset_partial_default_chat_template_no_drop_param(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "shards": 10,
                        "message_field_role": "from",
                        "message_field_content": "value",
                    }
                ],
            }
        )

        checked_cfg = validate_config(cfg)

        self._check_dataset(
            checked_cfg,
            cfg,
            expect_global_chat_template=ChatTemplate.chatml,
            expect_dataset_chat_template=ChatTemplate.tokenizer_default,
        )

        checked_cfg = validate_config(
            cfg,
            capabilities={
                "bf16": "false",
                "n_gpu": 1,
                "compute_capability": "8.0",
            },
            env_capabilities={
                "torch_version": "2.5.1",
            },
        )

        self._check_dataset(
            checked_cfg,
            cfg,
            expect_global_chat_template=ChatTemplate.chatml,
            expect_dataset_chat_template=ChatTemplate.tokenizer_default,
        )

    def test_dataset_chatml_chat_template_no_drop_param(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "chat_template",
                        "chat_template": "gemma",
                        "field_messages": "conversations",
                        "shards": 10,
                        "message_field_role": "from",
                        "message_field_content": "value",
                    }
                ],
            }
        )

        checked_cfg = validate_config(cfg)

        self._check_dataset(
            checked_cfg,
            cfg,
            expect_global_chat_template=cfg.chat_template,
            expect_dataset_chat_template=cfg.datasets[0].chat_template,
        )

        checked_cfg = validate_config(
            cfg,
            capabilities={
                "bf16": "false",
                "n_gpu": 1,
                "compute_capability": "8.0",
            },
            env_capabilities={
                "torch_version": "2.5.1",
            },
        )

        self._check_dataset(
            checked_cfg,
            cfg,
            expect_global_chat_template=cfg.chat_template,
            expect_dataset_chat_template=cfg.datasets[0].chat_template,
        )

    def test_dataset_sharegpt_deprecation(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "LDJnr/Puffin",
                        "type": "sharegpt",
                        "conversation": "chatml",
                    }
                ],
            }
        )

        # Check sharegpt deprecation is raised
        with pytest.raises(ValueError, match=r".*type: sharegpt.*` is deprecated.*"):
            validate_config(cfg)

        # Check that deprecation is not thrown for non-str type
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": {
                            "field_instruction": "instruction",
                            "field_output": "output",
                            "field_system": "system",
                            "format": "<|user|> {instruction} {input} <|model|>",
                            "no_input_format": "<|user|> {instruction} <|model|>",
                            "system_prompt": "",
                        },
                    }
                ],
            }
        )

        validate_config(cfg)

        # Check that deprecation is not thrown for non-sharegpt type
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
            }
        )

        validate_config(cfg)

    def test_message_property_mappings(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                        "message_property_mappings": {
                            "role": "role",
                            "content": "content",
                        },
                    }
                ],
            }
        )

        validate_config(cfg)


class TestOptimizerValidation(BaseValidation):
    """
    Test muon optimizer validation
    """

    def test_muon_deepspeed(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "optimizer": "muon",
                "deepspeed": "deepspeed_configs/zero3.json",
            }
        )

        with pytest.raises(ValueError, match=r".*is currently incompatible with*"):
            validate_config(cfg)

    def test_muon_fsdp(self, minimal_cfg):
        cfg = DictDefault(
            minimal_cfg
            | {
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    }
                ],
                "optimizer": "muon",
                "fsdp": ["full_shard"],
                "fsdp_config": {
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                },
            }
        )

        with pytest.raises(ValueError, match=r".*is currently incompatible with*"):
            validate_config(cfg)
