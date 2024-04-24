import torch, gc
from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

SELF_ATTN_WEIGHTS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_WEIGHTS = ["gate_proj", "up_proj", "down_proj"]

NUM_PROMPTS = 8
MAX_SEQ_LEN = 512

def per_tensor_quantize(tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Quantize a tensor using per-tensor static scaling factor.

    Args:
        tensor: The input tensor.
    """
    finfo = torch.finfo(torch.float8_e4m3fn)
    # Calculate the scale as dtype max divided by absmax.
    # Since .abs() creates a new tensor, we use aminmax to get
    # the min and max first and then calculate the absmax.
    min_val, max_val = tensor.aminmax()
    amax = min_val.abs().max(max_val.abs())
    scale = finfo.max / amax.clamp(min=1e-12)
    # scale and clamp the tensor to bring it to
    # the representative range of float8 data type
    # (as default cast is unsaturated)
    qweight = (tensor * scale).clamp(min=finfo.min, max=finfo.max)
    # Return both float8 data and the inverse scale (as float),
    # as both required as inputs to torch._scaled_mm
    qweight = qweight.to(torch.float8_e4m3fn)
    scale = scale.float().reciprocal()
    return qweight, scale


class FP8StaticLinearQuantizer(torch.nn.Module):
    def __init__(self, qweight, weight_scale):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.act_scale = None
    
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        # Dynamically quantize
        qinput, x_act_scale = per_tensor_quantize(x)

        # Update scale if needed.
        if self.act_scale is None:
            self.act_scale = torch.nn.Parameter(x_act_scale)
        elif x_act_scale > self.act_scale:
            self.act_scale = torch.nn.Parameter(x_act_scale)
        
        # Pass quantized to next layer so it has realistic data.
        output, _ = torch._scaled_mm(
            qinput,
            self.weight.t(),
            out_dtype=x.dtype,
            scale_a=self.act_scale,
            scale_b=self.weight_scale,
            bias=None,
        )
        return output.reshape(shape[0], shape[1], -1)


class FP8StaticLinear(torch.nn.Module):
    def __init__(self, qweight, weight_scale, act_scale=0.):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        self.act_scale = torch.nn.Parameter(act_scale, requires_grad=False)
    
    def per_tensor_quantize(self, tensor: torch.Tensor, inv_scale: float) -> torch.Tensor:
        # Scale and clamp the tensor to bring it to
        # the representative range of float8 data type
        # (as default cast is unsaturated)
        finfo = torch.finfo(torch.float8_e4m3fn)
        qweight = (tensor / inv_scale).clamp(min=finfo.min, max=finfo.max)
        return qweight.to(torch.float8_e4m3fn)

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])

        qinput = self.per_tensor_quantize(x, inv_scale=self.act_scale)
        output, _ = torch._scaled_mm(
            qinput,
            self.weight.t(),
            out_dtype=x.dtype,
            scale_a=self.act_scale,
            scale_b=self.weight_scale,
            bias=None,
        )
        return output.reshape(shape[0], shape[1], -1)


class FP8DynamicLinear(torch.nn.Module):
    def __init__(self, qweight, scale):
        super().__init__()
        self.weight = torch.nn.Parameter(qweight, requires_grad=False)
        self.weight_scale = torch.nn.Parameter(scale, requires_grad=False)
    
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        qinput, x_scale = per_tensor_quantize(x)
        
        output, _ = torch._scaled_mm(
            qinput,
            self.weight.t(),
            out_dtype=x.dtype,
            scale_a=x_scale,
            scale_b=self.weight_scale,
            bias=None,
        )
        return output.reshape(shape[0], shape[1], -1)


def quantize_proj(module, proj_name):
    proj = getattr(module, proj_name)
    quant_weight, quant_scale = per_tensor_quantize(proj.weight)
    quant_proj = FP8DynamicLinear(quant_weight, quant_scale)
    
    del proj
    setattr(module, proj_name, quant_proj)

def quantize_weights(model):
    for layer in model.model.layers:
        for proj_name in SELF_ATTN_WEIGHTS:
            quantize_proj(layer.self_attn, proj_name)
        for proj_name in MLP_WEIGHTS:
            quantize_proj(layer.mlp, proj_name)

def replace_dynamic_proj_w_quantizer(module, proj_name):
    dynamic_quant_proj = getattr(module, proj_name)
    assert isinstance(dynamic_quant_proj, FP8DynamicLinear)

    quantizer = FP8StaticLinearQuantizer(
        dynamic_quant_proj.weight, dynamic_quant_proj.weight_scale)
    del dynamic_quant_proj

    setattr(module, proj_name, quantizer)

def replace_quantizer_with_static_proj(module, proj_name):
    quantizer = getattr(module, proj_name)
    assert isinstance(quantizer, FP8StaticLinearQuantizer)

    static_proj = FP8StaticLinear(
        quantizer.weight, quantizer.weight_scale, quantizer.act_scale)
    del quantizer

    setattr(module, proj_name, static_proj)

def quantize_activations(model, calibration_tokens):
    # Replace layers with quantizer.
    for layer in model.model.layers:
        for proj_name in SELF_ATTN_WEIGHTS:
            replace_dynamic_proj_w_quantizer(layer.self_attn, proj_name)
        for proj_name in MLP_WEIGHTS:
            replace_dynamic_proj_w_quantizer(layer.mlp, proj_name)
    
    # Calibration.
    for row_idx in range(calibration_tokens.shape[0]):
        _ = model(calibration_tokens[row_idx].reshape(1,-1))

    # Replace quantizer with StaticLayer.
    for layer in model.model.layers:
        for proj_name in SELF_ATTN_WEIGHTS:
            replace_quantizer_with_static_proj(layer.self_attn, proj_name)
        for proj_name in MLP_WEIGHTS:
            replace_quantizer_with_static_proj(layer.mlp, proj_name)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sample_input_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is your name?" }],
        return_tensors="pt"
    ).to("cuda")

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=42).select(range(NUM_PROMPTS))
    ds = ds.map(lambda batch: {"text": tokenizer.apply_chat_template(batch["messages"], tokenize=False)})
    tokenizer.pad_token_id = tokenizer.eos_token_id
    calibration_tokens = tokenizer(
        ds["text"], 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_SEQ_LEN, 
        add_special_tokens=False).input_ids
    print(calibration_tokens.shape)

    # Quantize weights.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    quantize_weights(model)
    output = model.generate(input_ids=sample_input_tokens, max_new_tokens=20)
    print(tokenizer.decode(output[0]))

    # Quantize activations.
    quantize_activations(model, calibration_tokens=calibration_tokens.to("cuda"))
    output = model.generate(input_ids=sample_input_tokens, max_new_tokens=20)
    print(tokenizer.decode(output[0]))

    model.save_pretrained("mistral-fp-static-quant")
    tokenizer.save_pretrained("mistral-fp-static-quant")
