import importlib
import inspect
from abc import ABC, abstractmethod

import torch
from packaging import version

from ..modeling_utils import is_accelerate_available
from . import (is_auto_gptq_available, is_bitsandbytes_available,
               is_optimum_available, logging)
from .quantization_config import (BitsAndBytesConfig, GPTQConfig,
                                  QuantizationMethod)

logger = logging.get_logger(__name__)


def detect_model_quant(
    kwargs,
    config,
    quantization_config=None,
):
    """
    detect quant type
    consider not using kwargs, only Q-config

    """
    load_in_8bit = kwargs.pop("load_in_8bit", False)  # move to detect_model_quant
    load_in_4bit = kwargs.pop("load_in_4bit", False)  # move to detect_model_quant
    # TODO: consider deprecation notice for quant keargs outside quantization_config

    # default results
    quantization_method_from_args = None
    quantization_method_from_config = None
    # quantization_method = None
    quantizer = None

    # detect Q_method ========================================================================

    # handling args ----------------------------------------------------------------
    if quantization_config is not None:
        # BNB used as default method if Q config present but has no quant_method  # TODO consider raising error
        quantization_method_from_args = getattr(quantization_config, "quant_method", QuantizationMethod.BITS_AND_BYTES)

    elif quantization_config is None:
        if load_in_8bit or load_in_4bit:
            quantization_method_from_args = QuantizationMethod.BITS_AND_BYTES
            quantization_config = BitsAndBytesConfig.from_dict(
                config_dict={"load_in_8bit": load_in_8bit, "load_in_4bit": load_in_4bit},
                return_unused_kwargs=False,
                **kwargs,
            )

    # handling model config (and conflicts with config from args) ----------------------------

    if hasattr(config, "quantization_config"):
        quantization_method_from_config = config.quantization_config.get(
            "quant_method", QuantizationMethod.BITS_AND_BYTES
        )

        if quantization_method_from_args is not None:
            if quantization_method_from_config != QuantizationMethod.GPTQ:
                # By default quantization_method_from_args wins
                logger.warning(
                    "You passed `quantization_config` or equivalent parameters to `from_pretrained` but the model you're loading already has a"
                    " `quantization_config` attribute. The `quantization_config` attribute will be overwritten with the"
                    " one you passed to `from_pretrained`."
                )
                quantization_method = quantization_method_from_args
                config_origin = "args"
            else:
                # special case fror GPTQ config collision   # TODO consider unification
                loading_attr_dict = quantization_config.get_loading_attributes()
                for attr, val in loading_attr_dict.items():
                    config.quantization_config[attr] = val

                quantization_config = config.quantization_config
                quantization_method = quantization_method_from_config
                config_origin = "config"
                logger.warning(
                    "You passed `quantization_config` to `from_pretrained` but the model you're loading already has a "
                    "`quantization_config` attribute and has already quantized weights. However, loading attributes"
                    " (e.g. disable_exllama, use_cuda_fp16, max_input_length) will be overwritten with the one you passed to `from_pretrained`. The rest will be ignored."
                )

        else:
            quantization_method_from_config = config.quantization_config.get(
                "quant_method", QuantizationMethod.BITS_AND_BYTES
            )
            quantization_config = config.quantization_config
            quantization_method = quantization_method_from_config
            config_origin = "config"
    else:
        quantization_method = quantization_method_from_args
        config_origin = "args"

    # initialize quantizer ========================================================================

    if quantization_method == QuantizationMethod.BITS_AND_BYTES:
        quantization_config_kwargs = {
            k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters
        }
        # TODO: update quantization_config with quantization_config_kwargs

        if (load_in_8bit or load_in_4bit) and (len(quantization_config_kwargs) > 0):
            raise ValueError(
                "You can't pass `load_in_8bit` or any other `BitsAndBytesConfig` argument as a kwarg when passing "
                "`quantization_config` argument at the same time."
            )

        if quantization_config.load_in_4bit:
            quantizer = BnbFourBitHFQuantizer(quantization_config, quantization_config_kwargs)
        elif quantization_config.load_in_8bit:
            quantizer = BnbEightBitHFQuantizer(quantization_config, quantization_config_kwargs)

    elif quantization_method == QuantizationMethod.GPTQ:
        quantizer = GPTQHFQuantizer(quantization_config)
    else:
        raise NotImplementedError

    quantizer.config_origin = config_origin

    return quantization_method, quantizer


class HFQuantizer(ABC):
    config_origin = None  # can be "args" of "config"

    def __init__(self):
        self._validate_environment()
        pass

    @abstractmethod
    def _validate_environment(self, *args, **kwargs):
        pass

    def process_model_before_weight_loading(self, model):
        return model

    def process_model_after_weight_loading(self, model):
        model.is_loaded_in_4bit = False
        model.is_loaded_in_8bit = True

    @abstractmethod
    def load_model(self):
        pass

    def is_model_serializeable(self, model):
        pass

    def is_model_trainable(self, model):
        pass
    
    def set_torch_dtype(self, torch_dtype):
        return torch_dtype

    def set_target_dtype(self, torch_dtype):
        return torch_dtype

    def get_special_dtypes_update(self, model, torch_dtype):
        return dict()

    def validate_device_map(self, device_map):
        pass


# @abstractmethod def post-loading
# @abstractmethod def pre-saving
# @abstractmethod fresh_quantize_model
# @abstractmethod fresh_quantize_paramter


class GPTQHFQuantizer(HFQuantizer):

    """
    structure: assumes that layers are converted into GPTQ layers.
    storage: stores per weight .qweight, .qzeros, .scales
    saving: from state_dict() as any normal model
    loading: preprocess model into special layers, then load into state_dict as normal model
    """

    require_low_cpu_mem_usage = True

    def __init__(self, quantization_config):
        super().__init__
        from optimum.gptq import GPTQQuantizer

        self.quantizer = GPTQQuantizer.from_dict(quantization_config.to_dict())
        self.quantization_config = GPTQConfig.from_dict(quantization_config)

    def _validate_environment(self, torch_dtype, from_tf, from_flax):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to quantize or run quantize model.")

        if not (is_optimum_available() and is_auto_gptq_available()):
            raise ImportError(
                "Loading a GPTQ quantized model requires optimum (`pip install optimum`) and auto-gptq library (`pip install auto-gptq`)"
            )

        if version.parse(importlib.metadata.version("auto_gptq")) < version.parse("0.4.2"):
            raise ImportError(
                "You need a version of auto_gptq >= 0.4.2 to use GPTQ: `pip install --upgrade auto-gptq`"
            )

        if torch_dtype is None:
            torch_dtype = torch.float16
        else:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with GPTQ.")
            
    def set_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            torch_dtype = torch.float16
        elif torch_dtype != torch.float16:
            logger.info("We suggest you to set `torch_dtype=torch.float16` for better efficiency with GPTQ.")
        return torch_dtype

    def process_model_before_weight_loading(self, model):
        if self.config_origin == "config":
            model = self.quantizer.convert_model(model)
            model._is_quantized_training_enabled = True

        return model

    def process_model_after_weight_loading(self, model):
        super().process_model_after_weight_loading(model)

        if self.config_origin == "args":
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path
            if model.__class__.main_input_name != "input_ids":
                raise RuntimeError("We can only quantize pure text model.")
            self.quantizer.quantize_model(model, self.quantization_config.tokenizer)
            model.config.quantization_config = GPTQConfig.from_dict(self.quantizer.to_dict())
            model._is_quantized_training_enabled = True

        if self.config_origin == "config":
            model = self.quantizer.post_init_model(model)

        return model


class BnbHFQuantizer(HFQuantizer):
    
    """
    structure: 
    storage: 
    saving: 
    loading: 
    """

    use_keep_in_fp32_modules = True
    require_low_cpu_mem_usage = True

    def __init__(self, quantization_config, quantization_config_kwargs):
        super().__init__()
        if isinstance(quantization_config, dict):
            self.quantization_config = BitsAndBytesConfig.from_dict(quantization_config, return_unused_kwargs=False)
        elif isinstance(quantization_config, BitsAndBytesConfig):
            self.quantization_config = quantization_config
        else:
            raise ValueError(
                f"Invalid type for `quantization_config`: {type(quantization_config)}. Should be a `dict` or a"
                " `BitsAndBytesConfig` instance."
            )

    def _validate_environment(self, torch_dtype, from_tf, from_flax):
        if not (is_accelerate_available() and is_bitsandbytes_available()):
            raise ImportError(
                "Using `load_in_8bit=True` requires Accelerate: `pip install accelerate` and the latest version of"
                " bitsandbytes `pip install -i https://test.pypi.org/simple/ bitsandbytes` or"
                " pip install bitsandbytes` "
            )

        if torch_dtype is None:
            # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                f"Overriding torch_dtype={torch_dtype} with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning."
            )

        if from_tf or from_flax:
            raise ValueError(
                "Converting into 4-bit or 8-bit weights from tf/flax weights is currently not supported, please make"
                " sure the weights are in PyTorch format."
            )

    def set_torch_dtype(self, torch_dtype):
        if torch_dtype is None:
            # # We force the `dtype` to be float16, this is a requirement from `bitsandbytes`
            logger.info(
                f"Overriding torch_dtype={torch_dtype} with `torch_dtype=torch.float16` due to "
                "requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit. "
                "Pass your own torch_dtype to specify the dtype of the remaining non-linear layers or pass"
                " torch_dtype=torch.float16 to remove this warning."
            )
            torch_dtype = torch.float16
        return torch_dtype

    def get_special_dtypes_update(self, model, torch_dtype):
        return {
            name: torch_dtype
            for name, _ in model.named_parameters()
            if any(m in name for m in self.modules_to_not_convert)
        }

    def validate_device_map(self, device_map):
        device_map_without_lm_head = {
            key: device_map[key] for key in device_map.keys() if key not in self.modules_to_not_convert
        }
        if "cpu" in device_map_without_lm_head.values() or "disk" in device_map_without_lm_head.values():
            raise ValueError(
                """
                Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit
                the quantized model. If you want to dispatch the model on the CPU or the disk while keeping
                these modules in 32-bit, you need to set `load_in_8bit_fp32_cpu_offload=True` and pass a custom
                `device_map` to `from_pretrained`. Check
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu
                for more details.
                """
            )


class BnbEightBitHFQuantizer(BnbHFQuantizer):
    def _validate_environment(self):
        super()._validate_environment()

        if version.parse(importlib.metadata.version("bitsandbytes")) > version.parse("0.37.2"):
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 8bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

    def process_model_before_weight_loading(self, model, device_map, torch_dtype):
        keep_in_fp32_modules = model._keep_in_fp32_modules

        from ..integrations import (get_keys_to_not_convert,
                                    replace_with_bnb_linear)

        llm_int8_skip_modules = self.quantization_config.llm_int8_skip_modules
        load_in_8bit_fp32_cpu_offload = self.quantization_config.llm_int8_enable_fp32_cpu_offload

        logger.info("Activating loading for this model using {self.__class__}")

        # We keep some modules such as the lm_head in their original dtype for numerical stability reasons
        if llm_int8_skip_modules is None:
            self.modules_to_not_convert = get_keys_to_not_convert(model)
        else:
            self.modules_to_not_convert = llm_int8_skip_modules

        if not isinstance(self.modules_to_not_convert, list):
            self.modules_to_not_convert = [self.modules_to_not_convert]

        self.modules_to_not_convert.extend(keep_in_fp32_modules)

        # Extend the modules to not convert to keys that are supposed to be offloaded to `cpu` or `disk`
        if isinstance(device_map, dict) and len(device_map.keys()) > 1:
            keys_on_cpu = [key for key, value in device_map.items() if value in ["disk", "cpu"]]

            if len(keys_on_cpu) > 0 and not load_in_8bit_fp32_cpu_offload:
                raise ValueError(
                    "If you want to offload some keys to `cpu` or `disk`, you need to set "
                    "`llm_int8_enable_fp32_cpu_offload=True`. Note that these modules will not be "
                    " converted to 8-bit but kept in 32-bit."
                )
            self.modules_to_not_convert.extend(keys_on_cpu)

        model = replace_with_bnb_linear(
            model, modules_to_not_convert=self.modules_to_not_convert, quantization_config=self.quantization_config
        )
        # training in 8-bit is only available in 0.37.0+
        model._is_quantized_training_enabled = self.is_model_trainable()

        model.config.quantization_config = self.quantization_config
        model.is_8bit_serializable = self.is_model_serializeable()

        if torch_dtype is None:
            logger.warning(
                "You are loading your model in 8bit but you did not specify a `torch_dtype` attribute."
                "All non-linear modules will be loaded in full precision."
                " If you want to load the other modules in other precision, please specify a `torch_dtype` attribute."
            )

        if isinstance(device_map, str):
            special_dtypes = {}
            special_dtypes.update(
                {
                    name: torch_dtype
                    for name, _ in model.named_parameters()
                    if any(m in name for m in self.modules_to_not_convert)
                }
            )

        return model

    def set_target_dtype(self, torch_dtype):
        return torch.int8

    def is_model_serializeable(self, model=None):
        return version.parse(importlib.metadata.version("bitsandbytes")) > version.parse("0.37.2")

    def is_model_trainable(self, model=None):
        return version.parse(importlib.metadata.version("bitsandbytes")) >= version.parse("0.37.0")

    def process_model_after_weight_loading(self, model):
        super().process_model_after_weight_loading(model)
        model.is_loaded_in_8bit = True


class BnbFourBitHFQuantizer(BnbHFQuantizer):
    def _validate_environment(self):
        super()._validate_environment()

        if version.parse(importlib.metadata.version("bitsandbytes")) < version.parse("0.39.0"):
            raise ValueError(
                "You have a version of `bitsandbytes` that is not compatible with 4bit inference and training"
                " make sure you have the latest version of `bitsandbytes` installed"
            )

    def set_target_dtype(self, torch_dtype):
        if version.parse(importlib.metadata.version("accelerate")) > version.parse("0.19.0"):
            from accelerate.utils import CustomDtype

            return CustomDtype.INT4
        else:
            raise ValueError(
                "You are using `device_map='auto'` on a 4bit loaded version of the model. To automatically compute"
                " the appropriate device map, you should upgrade your `accelerate` library,"
                "`pip install --upgrade accelerate` or install it from source to support fp4 auto device map"
                "calculation. You may encounter unexpected behavior, or pass your own device map"
            )

    def process_model_after_weight_loading(self, model):
        super().process_model_after_weight_loading(model)
        model.is_loaded_in_4bit = True
