#!/usr/bin/env python3
"""
One-shot patch of modeling_esm.py for transformers 5.x compatibility.
Run on the Minerva login node (has internet). Fetches a clean original
from HuggingFace and applies all necessary fixes in one pass.
"""
import urllib.request
import sys
import os

TARGET = (
    "/sc/arion/work/cardia04/.cache/huggingface/modules/transformers_modules/"
    "InstaDeepAI/nucleotide_hyphen_transformer_hyphen_v2_hyphen_500m_hyphen_multi_hyphen_species/"
    "06615c1660c892fc199840c18123f8385b3542a8/modeling_esm.py"
)

URL = (
    "https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    "/resolve/main/modeling_esm.py"
)

print("Downloading clean original from HuggingFace...")
with urllib.request.urlopen(URL) as r:
    src = r.read().decode("utf-8")
print(f"  {len(src.splitlines())} lines downloaded")

changes = 0


def replace_once(s, old, new, label):
    global changes
    if old not in s:
        print(f"  [WARN] pattern not found: {label}")
        return s
    count = s.count(old)
    if count > 1:
        print(f"  [WARN] pattern found {count} times (expected 1): {label}")
    result = s.replace(old, new, 1)
    changes += 1
    print(f"  [OK] {label}")
    return result


# ---------------------------------------------------------------------------
# Patch 1 – transformers.file_utils removed in v5
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    'from transformers.file_utils import (\n    add_code_sample_docstrings,\n    add_start_docstrings,\n    add_start_docstrings_to_model_forward,\n)',
    '''try:
    from transformers.file_utils import (
        add_code_sample_docstrings,
        add_start_docstrings,
        add_start_docstrings_to_model_forward,
    )
except ImportError:
    try:
        from transformers.utils import (
            add_code_sample_docstrings,
            add_start_docstrings,
            add_start_docstrings_to_model_forward,
        )
    except ImportError:
        def add_start_docstrings(*args, **kwargs):
            def decorator(fn): return fn
            return decorator
        def add_start_docstrings_to_model_forward(*args, **kwargs):
            def decorator(fn): return fn
            return decorator
        def add_code_sample_docstrings(*args, **kwargs):
            def decorator(fn): return fn
            return decorator''',
    "file_utils import fallback",
)

# ---------------------------------------------------------------------------
# Patch 2 – find_pruneable_heads_and_indices / prune_linear_layer removed in v5
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    "from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer",
    '''try:
    from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
except ImportError:
    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        return heads, index

    def prune_linear_layer(layer, index, dim=0):
        index = index.to(layer.weight.device)
        W = layer.weight.index_select(dim, index).clone().detach()
        if layer.bias is not None:
            b = (layer.bias.clone().detach() if dim == 1
                 else layer.bias[index].clone().detach())
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = nn.Linear(new_size[1], new_size[0],
                              bias=(layer.bias is not None)).to(layer.weight.device)
        new_layer.weight.requires_grad = False
        new_layer.weight.copy_(W.contiguous())
        new_layer.weight.requires_grad = True
        if layer.bias is not None:
            new_layer.bias.requires_grad = False
            new_layer.bias.copy_(b.contiguous())
            new_layer.bias.requires_grad = True
        return new_layer''',
    "pytorch_utils import fallback with local implementations",
)

# ---------------------------------------------------------------------------
# Patch 3 – EsmSelfAttention.__init__: config.is_decoder default removed
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    "        self.is_decoder = config.is_decoder\n\n    def transpose_for_scores",
    "        self.is_decoder = getattr(config, 'is_decoder', False)\n\n    def transpose_for_scores",
    "EsmSelfAttention.is_decoder",
)

# ---------------------------------------------------------------------------
# Patch 4 – EsmLayer.__init__: three config attributes with removed defaults
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    "        self.chunk_size_feed_forward = config.chunk_size_feed_forward",
    "        self.chunk_size_feed_forward = getattr(config, 'chunk_size_feed_forward', 0)",
    "EsmLayer.chunk_size_feed_forward",
)
src = replace_once(
    src,
    "        self.is_decoder = config.is_decoder\n        self.add_cross_attention = config.add_cross_attention",
    "        self.is_decoder = getattr(config, 'is_decoder', False)\n        self.add_cross_attention = getattr(config, 'add_cross_attention', False)",
    "EsmLayer.is_decoder + add_cross_attention",
)

# ---------------------------------------------------------------------------
# Patch 5 – EsmEncoder.forward: self.config.add_cross_attention (two spots)
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    "            () if output_attentions and self.config.add_cross_attention else None",
    "            () if output_attentions and getattr(self.config, 'add_cross_attention', False) else None",
    "EsmEncoder all_cross_attentions init",
)
src = replace_once(
    src,
    "                if self.config.add_cross_attention:",
    "                if getattr(self.config, 'add_cross_attention', False):",
    "EsmEncoder output_attentions cross_attention check",
)

# ---------------------------------------------------------------------------
# Patch 6 – EsmModel.forward: self.config.is_decoder (two spots)
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    "        if self.config.is_decoder:\n            use_cache = use_cache if use_cache is not None else self.config.use_cache",
    "        if getattr(self.config, 'is_decoder', False):\n            use_cache = use_cache if use_cache is not None else self.config.use_cache",
    "EsmModel.forward is_decoder use_cache",
)
src = replace_once(
    src,
    "        if self.config.is_decoder and encoder_hidden_states is not None:",
    "        if getattr(self.config, 'is_decoder', False) and encoder_hidden_states is not None:",
    "EsmModel.forward is_decoder cross-attention",
)

# ---------------------------------------------------------------------------
# Patch 7 – EsmForMaskedLM.__init__: config.is_decoder default removed
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    '        if config.is_decoder:\n            logger.warning(\n                "If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for "',
    '        if getattr(config, \'is_decoder\', False):\n            logger.warning(\n                "If you want to use `EsmForMaskedLM` make sure `config.is_decoder=False` for "',
    "EsmForMaskedLM.__init__ is_decoder check",
)

# ---------------------------------------------------------------------------
# Patch 8 – EsmForMaskedLM: add all_tied_weights_keys property (v5 expects dict)
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    'class EsmForMaskedLM(EsmPreTrainedModel):\n    _tied_weights_keys = ["lm_head.decoder.weight"]',
    '''class EsmForMaskedLM(EsmPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight"]

    @property
    def all_tied_weights_keys(self):
        return {k: k for k in (self._tied_weights_keys or [])}''',
    "EsmForMaskedLM.all_tied_weights_keys property",
)

# ---------------------------------------------------------------------------
# Patch 9 – EsmPreTrainedModel: add get_head_mask (removed or moved in v5)
# ---------------------------------------------------------------------------
src = replace_once(
    src,
    "    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights\n    def _init_weights(self, module):",
    '''    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):''',
    "EsmPreTrainedModel.get_head_mask local implementation",
)

# ---------------------------------------------------------------------------
# Write result
# ---------------------------------------------------------------------------
print(f"\n{changes}/9 patches applied")
if changes < 9:
    print("WARNING: not all patches applied — check warnings above before continuing")
    sys.exit(1)

with open(TARGET, "w") as f:
    f.write(src)
print(f"Written to: {TARGET}")
print("Done.")
