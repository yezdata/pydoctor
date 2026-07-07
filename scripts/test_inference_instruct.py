import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM
from peft import PeftModel

from pydoctor_model_pipeline.utils.tokenizer import get_instruct_tokenizer
from pydoctor_model_pipeline.utils.config_models import MainConfig

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
config = MainConfig.from_yaml("configs_kaggle.yaml")


SYSTEM_PROMPT = f"""You are a professional Python docstring generator.
You will receive Python code block (function, class, class method).
Generate a concise docstring for the code block marked with {config.tokenizer.spec_tokens.docstring_placeholder_token}.
The style of the doctring have to be only sentences summarizing the code block.
Output ONLY the raw docstring text.
Do NOT include Args, Returns, Raises, or any structured sections.
Do NOT output any conversational text or explanations."""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def get_chat_template(user_prompt: str) -> str:
    final_output = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return final_output


tokenizer = get_instruct_tokenizer(config.tokenizer, hf_token=hf_token)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

base_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name, torch_dtype=DTYPE, device_map=DEVICE, token=hf_token
)

model = PeftModel.from_pretrained(
    base_model,
    "models/smollm2_1_7b_instruct/finetune_instruct/epoch_3/",
)
model.eval()


USER_INPUT_LIST = [
    '''
class DecoderLayer(nn.Module):
    """<docstring_place>"""

    def __init__(self, config: DecoderConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()

        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.rope = rope

        # Attention projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        self.ffn = SwiGLU(config.d_model, config.d_ffn)

        self.dropout = nn.Dropout(config.dropout)

        # mark for initialization
        self.out_proj._is_residual = True
        self.ffn.wo._is_residual = True

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass''',
    '''
class DecoderLayer(nn.Module):

    def __init__(self, config: DecoderConfig, rope: RotaryPositionalEmbeddings):
        super().__init__()

        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head
        self.rope = rope

        # Attention projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        self.ffn = SwiGLU(config.d_model, config.d_ffn)

        self.dropout = nn.Dropout(config.dropout)

        # mark for initialization
        self.out_proj._is_residual = True
        self.ffn.wo._is_residual = True

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """<docstring_place>"""
        # MULTI-HEAD ATTENTION
        residual = x
        nx = self.norm1(x)
        B, S, _ = nx.shape

        # Projections -> (B, S, N_heads, D_head)
        q = self.q_proj(nx).view(B, S, self.n_head, self.d_head)
        k = self.k_proj(nx).view(B, S, self.n_head, self.d_head)
        v = self.v_proj(nx).view(B, S, self.n_head, self.d_head)

        q = self.rope(q, input_pos=input_pos)
        k = self.rope(k, input_pos=input_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True if mask is None else False,
            attn_mask=mask,
            dropout_p=self.dropout.p if self.dropout.training else 0.0,
        )

        # Join heads -> (B, S, D_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        x = residual + self.dropout(self.out_proj(attn_out))

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x''',
    '''
class DecoderModel(nn.Module):
    """<docstring_place>"""

    def __init__(
        self,
        config: DecoderConfig,
        eos_token_id: int,
        expected_max_seq_len: int = 4096,
    ):
        super().__init__()
        self.eos_token_id = eos_token_id

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.final_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_embedding.weight

        self.rope = RotaryPositionalEmbeddings(
            dim=config.d_model // config.n_head, max_seq_len=expected_max_seq_len
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(config, self.rope) for _ in range(config.n_layers)]
        )

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        pass

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass''',
    '''
class DecoderModel(nn.Module):

    def __init__(
        self,
        config: DecoderConfig,
        eos_token_id: int,
        expected_max_seq_len: int = 4096,
    ):
        super().__init__()
        self.eos_token_id = eos_token_id

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.final_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_embedding.weight

        self.rope = RotaryPositionalEmbeddings(
            dim=config.d_model // config.n_head, max_seq_len=expected_max_seq_len
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(config, self.rope) for _ in range(config.n_layers)]
        )

    def resize_token_embeddings(self, new_vocab_size: int) -> None:
        """<docstring_place>"""
        old_embeddings = self.token_embedding

        if new_vocab_size == old_embeddings.num_embeddings:
            return

        assert new_vocab_size > old_embeddings.num_embeddings

        new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.embedding_dim)
        init_weights_modern(new_embeddings)

        with torch.no_grad():
            new_embeddings.weight.data[: old_embeddings.num_embeddings] = (
                old_embeddings.weight.data
            )
        self.token_embedding = new_embeddings

        new_head = nn.Linear(old_embeddings.embedding_dim, new_vocab_size, bias=False)
        new_head.weight = self.token_embedding.weight
        self.head = new_head''',
    '''
class DecoderModel(nn.Module):

    def __init__(
        self,
        config: DecoderConfig,
        eos_token_id: int,
        expected_max_seq_len: int = 4096,
    ):
        super().__init__()
        self.eos_token_id = eos_token_id

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.final_norm = RMSNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying
        self.head.weight = self.token_embedding.weight

        self.rope = RotaryPositionalEmbeddings(
            dim=config.d_model // config.n_head, max_seq_len=expected_max_seq_len
        )

        self.layers = nn.ModuleList(
            [DecoderLayer(config, self.rope) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """<docstring_place>"""

        x = self.token_embedding(x)

        for layer in self.layers:
            x = layer(x, mask, input_pos)

        x_norm = self.final_norm(x)

        return self.head(x_norm)
''',
]


for snippet in USER_INPUT_LIST:
    prompt = get_chat_template(snippet)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.encode("<|im_end|>"),
        )

    generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("\n", "---" * 20)
    print(output_text)
