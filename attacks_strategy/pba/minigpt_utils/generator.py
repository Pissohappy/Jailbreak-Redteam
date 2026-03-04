import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


import torch
import time


def diagnose_error(model, context_emb):
    """Helper function to check input and GPU state"""
    try:
        # Check dimensions
        if context_emb.dim() == 2:
            print("[Info] context_emb missing batch dim, auto unsqueeze.")
            context_emb = context_emb.unsqueeze(0)

        # Check dtype
        print(
            f"[Info] context_emb dtype: {context_emb.dtype}, device: {context_emb.device}"
        )
        # if context_emb.dtype != torch.float32:
        #     print("[Warning] Detected non-fp32 embedding — possible fp16 instability risk.")

        # Check NaN/Inf
        if not torch.isfinite(context_emb).all():
            raise ValueError(
                "[Error] Found NaN or Inf in context_emb before generation."
            )

        # Check GPU memory
        if torch.cuda.is_available():
            mem = torch.cuda.mem_get_info()
            free_mb = mem[0] / 1024**2
            print(f"[Info] Free GPU memory: {free_mb:.1f} MB")
            if free_mb < 2000:
                print("[Warning] GPU memory is very low — may cause hang or OOM.")
    except Exception as e:
        print(f"[Diagnose Error] {e}")

    return context_emb


class Generator:

    def __init__(
        self,
        model,
        max_new_tokens=100,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1,
        temperature=1.0,
        device="cuda:0",
    ):

        self.model = model
        self.device = device

        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.min_length = min_length
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.temperature = temperature

        stop_words_ids = [
            torch.tensor([835]).to(self.device),
            torch.tensor([2277, 29937]).to(self.device),
        ]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)]
        )

    # def generate(self, prompt):

    #     outputs = self.model.llama_model.generate(  # module
    #         inputs_embeds=prompt.context_embs[0],
    #         max_new_tokens=self.max_new_tokens,
    #         stopping_criteria=self.stopping_criteria,
    #         num_beams=self.num_beams,
    #         do_sample=True,
    #         min_length=self.min_length,
    #         top_p=self.top_p,
    #         repetition_penalty=self.repetition_penalty,
    #         length_penalty=self.length_penalty,
    #         temperature=self.temperature,
    #     )

    #     output_token = outputs[0]
    #     if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
    #         output_token = output_token[1:]
    #     if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
    #         output_token = output_token[1:]
    #     output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)  # module
    #     output_text = output_text.split('###')[0]  # remove the stop sign '###'
    #     output_text = output_text.split('Assistant:')[-1].strip()

    #     return output_text, output_token.cpu().numpy()
    def generate(self, prompt):
        """Improved generate: with error detection and exception diagnosis"""
        context_emb = prompt.context_embs[0]
        # context_emb = diagnose_error(self.model.llama_model, context_emb)

        if context_emb.dim() == 2:
            context_emb = context_emb.unsqueeze(0)

        try:
            start_t = time.time()
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model.llama_model.generate(
                    inputs_embeds=context_emb.half(),
                    max_new_tokens=min(self.max_new_tokens, 64),
                    stopping_criteria=self.stopping_criteria,
                    num_beams=self.num_beams,
                    do_sample=True,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    length_penalty=self.length_penalty,
                    temperature=self.temperature,
                    min_length=self.min_length,
                    max_time=10.0,
                )
            elapsed = time.time() - start_t
            print(f"[Info] Generation finished in {elapsed:.2f}s")
        except torch.cuda.OutOfMemoryError:
            print("[Error] CUDA OOM — reduce batch_size or max_new_tokens.")
            torch.cuda.empty_cache()
            return "CUDA_OOM", None
        except RuntimeError as e:
            if "NaN" in str(e) or "rsqrt" in str(e):
                print(
                    "[Error] Detected NaN instability (likely fp16 LayerNorm). Try FP32 precision."
                )
            elif "context" in str(e):
                print("[Error] Possibly too long prompt — check token length.")
            else:
                print(f"[RuntimeError] {e}")
            return "GENERATION_ERROR", None
        except Exception as e:
            print(f"[Unexpected Error] {e}")
            return "UNKNOWN_ERROR", None

        # Normal decoding part
        output_token = outputs[0]
        if output_token[0] in [0, 1]:
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(
            output_token, add_special_tokens=False
        )
        output_text = output_text.split("###")[0]
        output_text = output_text.split("Assistant:")[-1].strip()

        torch.cuda.empty_cache()
        return output_text, output_token.cpu().numpy()
