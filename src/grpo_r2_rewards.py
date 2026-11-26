"""Reward v2 for Swift generation, dataset parity and per-optimizer-step telemetry."""
import json

from src.grpo_reward import decode_rollout
from src.grpo_reward_v2 import score_completion_v2

MU = 0.5
COMPONENTS = ('mean_n_lost', 'mean_used_per_1000', 'invalid_rate', 'mean_add_count', 'multi_add_rate')


def score_raw(raw, state):
    return score_completion_v2(raw, state, MU)


def rollout_scores(tokenizer, completions, reward_state_json, response_token_ids):
    if not (len(completions) == len(reward_state_json) == len(response_token_ids)):
        raise ValueError('reward inputs and generated tokens must be aligned')
    scores = []
    for packed, tokens in zip(reward_state_json, response_token_ids):
        state = json.loads(packed)
        raw = decode_rollout(tokenizer, state['prompt_token_ids_tail'], tokens)
        scores.append(score_raw(raw, state))
    return scores


class RewardComponents:
    """Sample-weighted totals. Add counts use distinct requested IDs; malformed JSON contributes zero."""
    def __init__(self):
        self.n = self.lost = self.used = self.invalid = self.adds = self.multi = 0

    def observe(self, score):
        action = score['action']
        added = len(set(action['add'])) if action is not None else 0
        self.n += 1
        self.lost += score['n_lost']
        self.used += score['used_after']
        self.invalid += score['invalid_penalty']
        self.adds += added
        self.multi += added >= 2

    def result(self):
        if not self.n:
            raise ValueError('no plugin reward observations')
        return {'reward_component_n': self.n, 'mean_n_lost': self.lost/self.n,
                'mean_used_per_1000': self.used/(1000*self.n), 'invalid_rate': self.invalid/self.n,
                'mean_add_count': self.adds/self.n, 'multi_add_rate': self.multi/self.n}


def take_components(trainer_state):
    pending = getattr(trainer_state, '_ctx_grpo_r2_components', None)
    if pending is None or pending[0] != trainer_state.global_step or pending[1].n != 32:
        raise ValueError(f'missing/misaligned plugin rewards at step {trainer_state.global_step}')
    del trainer_state._ctx_grpo_r2_components
    return pending[1].result()


class CtxStepRewardV2:
    """Swift synchronous ORM. The same callable works on saved greedy token IDs during D."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer.backend_tokenizer

    def __call__(self, completions, reward_state_json, response_token_ids, trainer_state=None, **kwargs):
        scores = rollout_scores(self.tokenizer, completions, reward_state_json, response_token_ids)
        if trainer_state is not None:
            # Swift supplies self.state before the corresponding optimizer update.
            step = trainer_state.global_step + 1
            pending = getattr(trainer_state, '_ctx_grpo_r2_components', None)
            if pending is None:
                pending = step, RewardComponents()
                trainer_state._ctx_grpo_r2_components = pending
            if pending[0] != step:
                raise ValueError('unconsumed reward components from a prior optimizer step')
            for score in scores:
                pending[1].observe(score)
        return [score['reward'] for score in scores]

