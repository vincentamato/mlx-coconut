import mlx.core as mx
import mlx.nn as nn
from collections import namedtuple

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8

class CoconutModel(nn.Module):
    def __init__(self, base_causallm, latent_token_id, start_latent_id, end_latent_id, eos_token_id):
        super().__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.eos_token_id = eos_token_id
        self.embedding = self.base_causallm.get_input_embeddings()

    def __call__(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        logits = []
        latent_mask = (input_ids == self.latent_token_id)
        
        latent_lists = []
        for batch_idx in range(input_ids.shape[0]):
            batch_positions = []
            for seq_idx in range(input_ids.shape[1]):
                if latent_mask[batch_idx, seq_idx].item():
                    batch_positions.append(seq_idx)
            latent_lists.append(batch_positions)

        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0
        next_compute_range = (0, int(input_ids.shape[1]))
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            all_positions = [pos for batch_list in latent_lists for pos in batch_list]
            if all_positions:
                min_pos = int(min(all_positions))
                next_compute_range = (0, min_pos)

        kv_cache = None

        for pass_idx in range(max_n_latents):
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, next_compute_range[0]:next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0
            else:
                past_key_values = [
                    (k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                    for k, v in kv_cache
                ]
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)
            next_compute_range = (
                int(next_compute_range[1]),
                int(input_ids.shape[1] if pass_idx + 1 >= max_n_latents else next_compute_range[1] + 1)
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values
            
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            tensor_list = [
                [inputs_embeds[batch_idx, pos, :] for pos in range(inputs_embeds.shape[1])]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            inputs_embeds = mx.stack([
                mx.stack(tensor_list[batch_idx])
                for batch_idx in range(inputs_embeds.shape[0])
            ])

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[:, next_compute_range[0]:next_compute_range[1], :],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0]:next_compute_range[1]],
            past_key_values=(
                [(k[:, :, :next_compute_range[0], :], v[:, :, :next_compute_range[0], :])
                 for k, v in kv_cache] if kv_cache else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)
        self.gen_forward_cnt += max_n_latents + 1
        logits = mx.concatenate(logits, axis=-2)
        
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            loss = nn.losses.cross_entropy(
                shift_logits.reshape(-1, shift_logits.shape[-1]), 
                shift_labels.reshape(-1)
            ).mean()
        else:
            loss = None

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)
    
    def generate(self, input_ids, attention_mask, max_new_tokens=16, synced_gpus=False, **kwargs):
        """Generate method for text generation."""
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].tolist()
        labels = mx.array(input_ids)
        position_ids = mx.arange(0, int(input_ids.shape[1])).reshape(1, -1)
        
        outputs = self.__call__(input_ids, mx.ones_like(input_ids), labels, position_ids)
        inputs_embeds = outputs.inputs_embeds

        next_token = mx.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        
        new_token_embed = self.embedding(mx.array([next_token])).reshape(1, 1, -1)
        new_inputs_embeds = mx.concatenate([inputs_embeds, new_token_embed], axis=1)
        current_length = inputs_embeds.shape[1]
        
        for i in range(max_new_tokens - 1):
            pos_ids = mx.arange(0, current_length + 1 + i).reshape(1, -1)
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds, position_ids=pos_ids)
            
            self.gen_forward_cnt += 1
            next_token = mx.argmax(outputs.logits[0, -1]).item()
            
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(mx.array([next_token])).reshape(1, 1, -1)
            new_inputs_embeds = mx.concatenate([new_inputs_embeds, new_token_embed], axis=1)

        if synced_gpus:
            target_count = max_new_tokens + MAX_N_LATENT
            while self.gen_forward_cnt < target_count:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        return mx.array(tokens).reshape(1, -1)