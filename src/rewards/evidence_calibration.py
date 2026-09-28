"""
Evidence-calibrated reward:

- keep the current answer-level visual score as the base term
- add a relative answer margin against the original answer and same-step siblings
- use visual clues as evidence proposals rather than only an absolute reward
- verify candidates with answer-conditioned support views and masked counterfactuals

This is intentionally lightweight and training-free.
"""

import math

from evaluation.visual_clues import format_visual_clues
from prompts.clue_prompts import build_visual_clue_request_messages, build_visual_clue_support_messages
from rewards.crop_utils import build_candidate_crops, build_masked_view, build_view_media, get_primary_image, summarize_views
from rewards.visual_likelihood import extract_normalized_short_answer_for_scoring


def _clamp(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(value)))


class EvidenceCalibratedReward:
    def __init__(
        self,
        visual_reward,
        answer_weight=1.0,
        margin_weight=0.5,
        clue_weight=0.0,
        crop_weight=0.0,
        max_crop_candidates_per_step=2,
    ):
        self.visual_reward = visual_reward
        self.answer_weight = float(answer_weight)
        self.margin_weight = float(margin_weight)
        self.clue_weight = float(clue_weight)
        self.crop_weight = float(crop_weight)
        self.max_crop_candidates_per_step = max(int(max_crop_candidates_per_step), 0)

        reward_mode = getattr(visual_reward, "reward_mode", "likelihood") if visual_reward is not None else "likelihood"
        self.reward_mode = reward_mode
        self.answer_reward_center = 0.0 if reward_mode == "contrastive" else 0.5
        self.clue_reward_center = 0.0
        self.crop_reward_center = 0.0

    def _delta_temperature(self):
        if self.visual_reward is None:
            return 1.0
        return max(float(getattr(self.visual_reward, "contrastive_temperature", 1.0)), 1e-6)

    def _squash_delta(self, delta):
        return math.tanh(float(delta) / self._delta_temperature())

    @property
    def reward_center(self):
        return (
            self.answer_weight * self.answer_reward_center
            + self.clue_weight * self.clue_reward_center
            + self.crop_weight * self.crop_reward_center
        )

    def _score_answer_only(self, question, answer_text, media=None, problem=None):
        if self.visual_reward is None:
            return {
                "answer_score": 0.0,
                "answer_reward": 0.0,
                "visual_logprob": None,
                "prior_logprob": None,
            }

        raw_score, visual_logprob, prior_logprob = self.visual_reward.score_short_answer(
            answer_text,
            media=media,
            problem=problem,
        )

        if self.reward_mode == "contrastive":
            answer_reward = math.tanh(raw_score / self.visual_reward.contrastive_temperature)
        else:
            answer_reward = _clamp((raw_score + 2.0) / 2.0)

        return {
            "answer_score": float(raw_score),
            "answer_reward": float(answer_reward),
            "visual_logprob": visual_logprob,
            "prior_logprob": prior_logprob,
        }

    def _normalize_answer_display(self, answer_text, problem=None):
        normalized = extract_normalized_short_answer_for_scoring(answer_text, problem)
        if normalized is not None:
            return normalized
        return str(answer_text or "").strip()

    def _score_clues(self, answer_text, clues, media=None, problem=None):
        empty = {
            "clue_score": 0.0,
            "clue_reward": 0.0,
            "clue_visual_logprob": None,
            "clue_prior_logprob": None,
            "clue_text": None,
        }
        if self.visual_reward is None or not clues:
            return empty

        clue_text = format_visual_clues(clues)
        if not clue_text:
            return empty

        candidate_answer = self._normalize_answer_display(answer_text, problem=problem)
        messages = build_visual_clue_support_messages(
            problem=problem,
            media=media,
            candidate_answer=candidate_answer,
            num_clues=len(clues),
            include_image=True,
        )
        clue_visual_logprob = self.visual_reward.score_completion_logprob(
            prompt_messages=messages,
            target_text=clue_text,
            media=media,
            include_image=True,
        )
        clue_prior_logprob = None
        if self.reward_mode == "contrastive":
            prior_messages = build_visual_clue_support_messages(
                problem=problem,
                media=media,
                candidate_answer=candidate_answer,
                num_clues=len(clues),
                include_image=False,
            )
            clue_prior_logprob = self.visual_reward.score_completion_logprob(
                prompt_messages=prior_messages,
                target_text=clue_text,
                media=media,
                include_image=False,
            )
            clue_score = clue_visual_logprob - self.visual_reward.prior_weight * clue_prior_logprob
            clue_reward = math.tanh(clue_score / self.visual_reward.contrastive_temperature)
        else:
            clue_score = clue_visual_logprob
            clue_reward = _clamp((clue_score + 2.0) / 2.0)

        return {
            "clue_score": float(clue_score),
            "clue_reward": float(clue_reward),
            "clue_visual_logprob": clue_visual_logprob,
            "clue_prior_logprob": clue_prior_logprob,
            "clue_text": clue_text,
        }

    def _empty_clue_info(self):
        return {
            "clue_score": 0.0,
            "clue_reward": 0.0,
            "clue_margin_score": 0.0,
            "clue_visual_logprob": None,
            "clue_prior_logprob": None,
            "clue_text": None,
            "competitor_clue_baseline": 0.0,
        }

    def score_candidate_conditioned_clues(
        self,
        answer_texts,
        candidate_clues,
        media=None,
        problem=None,
        reference_answer=None,
        reference_clues=None,
    ):
        if answer_texts is None:
            return []

        answer_texts = list(answer_texts)
        candidate_clues = list(candidate_clues or [])
        if len(candidate_clues) < len(answer_texts):
            candidate_clues = candidate_clues + [[] for _ in range(len(answer_texts) - len(candidate_clues))]

        clue_infos = []
        for answer_text, clues in zip(answer_texts, candidate_clues):
            key = self._normalize_answer_display(answer_text, problem=problem)
            clue_infos.append(
                self._score_clues(
                    answer_text=key,
                    clues=clues,
                    media=media,
                    problem=problem,
                )
            )

        reference_clue_score = None
        reference_info = None
        if reference_answer is not None and reference_clues:
            reference_key = self._normalize_answer_display(reference_answer, problem=problem)
            reference_info = self._score_clues(
                answer_text=reference_key,
                clues=reference_clues,
                media=media,
                problem=problem,
            )
            reference_clue_score = reference_info["clue_score"]

        clue_scores = [info["clue_score"] for info in clue_infos]
        results = []
        for idx, info in enumerate(clue_infos):
            competitors = []
            if reference_clue_score is not None:
                competitors.append(reference_clue_score)
            for jdx, sibling_score in enumerate(clue_scores):
                if jdx == idx:
                    continue
                competitors.append(sibling_score)

            baseline = sum(competitors) / len(competitors) if competitors else info["clue_score"]
            margin_delta = info["clue_score"] - baseline
            margin_score = self._squash_delta(margin_delta)
            result = dict(info)
            result.update(
                {
                    "conditioned_clue_margin_delta": float(margin_delta),
                    "conditioned_clue_margin_score": float(margin_score),
                    "conditioned_competitor_clue_baseline": float(baseline),
                    "conditioned_reference_clue_score": None if reference_clue_score is None else float(reference_clue_score),
                    "conditioned_reference_clue_text": None if reference_info is None else reference_info.get("clue_text"),
                }
            )
            results.append(result)

        return results

    def _score_clue_grounding_media(self, clues, media=None, problem=None):
        empty = {
            "support_score": 0.0,
            "support_reward": 0.0,
            "support_visual_logprob": None,
            "support_prior_logprob": None,
            "clue_text": None,
        }
        if self.visual_reward is None or not clues:
            return empty

        clue_text = format_visual_clues(clues)
        if not clue_text:
            return empty

        messages = build_visual_clue_request_messages(
            problem=problem,
            media=media,
            num_clues=len(clues),
            draft_response=None,
            include_image=True,
        )
        support_visual_logprob = self.visual_reward.score_completion_logprob(
            prompt_messages=messages,
            target_text=clue_text,
            media=media,
            include_image=True,
        )
        support_prior_logprob = None
        if self.reward_mode == "contrastive":
            prior_messages = build_visual_clue_request_messages(
                problem=problem,
                media=media,
                num_clues=len(clues),
                draft_response=None,
                include_image=False,
            )
            support_prior_logprob = self.visual_reward.score_completion_logprob(
                prompt_messages=prior_messages,
                target_text=clue_text,
                media=media,
                include_image=False,
            )
            support_score = support_visual_logprob - self.visual_reward.prior_weight * support_prior_logprob
            support_reward = math.tanh(support_score / self.visual_reward.contrastive_temperature)
        else:
            support_score = support_visual_logprob
            support_reward = _clamp((support_score + 2.0) / 2.0)

        return {
            "support_score": float(support_score),
            "support_reward": float(support_reward),
            "support_visual_logprob": support_visual_logprob,
            "support_prior_logprob": support_prior_logprob,
            "clue_text": clue_text,
        }

    def prepare_crop_evidence(
        self,
        question,
        clues,
        media=None,
        problem=None,
        top_k_positive=2,
        max_negative=3,
        grid_sizes=(2, 3),
    ):
        if self.visual_reward is None or self.crop_weight <= 0 or not media:
            return None

        primary_image = get_primary_image(media)
        if primary_image is None:
            return None

        candidate_views = build_candidate_crops(primary_image, grid_sizes=grid_sizes, include_center=True)
        if not candidate_views:
            return None

        top_k_positive = max(int(top_k_positive), 0)
        max_negative = max(int(max_negative), 0)

        ranked_views = []
        if clues:
            for view in candidate_views:
                support_info = self._score_clue_grounding_media(
                    clues=clues,
                    media=view.get("media"),
                    problem=problem,
                )
                ranked_view = dict(view)
                ranked_view["support_score"] = support_info["support_score"]
                ranked_views.append(ranked_view)
            ranked_views.sort(key=lambda item: item.get("support_score", float("-inf")), reverse=True)
        else:
            ranked_views = [dict(view, support_score=0.0) for view in candidate_views]

        positive_views = ranked_views[:top_k_positive] if top_k_positive > 0 else []
        positive_ids = {view.get("view_id") for view in positive_views}
        negative_candidates = [view for view in ranked_views if view.get("view_id") not in positive_ids]
        negative_views = negative_candidates[-max_negative:] if max_negative > 0 else []

        crop_evidence = {
            "candidate_views": candidate_views,
            "proposal_positive_views": positive_views,
            "proposal_negative_views": negative_views,
            "top_k_positive": top_k_positive,
            "max_negative": max_negative,
            "summary": {
                "num_candidate_crops": len(candidate_views),
                "positive_views": summarize_views(positive_views),
                "negative_views": summarize_views(negative_views),
                "grid_sizes": [int(item) for item in grid_sizes],
            },
        }
        return crop_evidence

    def _empty_crop_info(self):
        return {
            "crop_evidence_score": 0.0,
            "crop_reward": 0.0,
            "positive_crop_scores": [],
            "negative_crop_scores": [],
            "masked_crop_scores": [],
            "positive_crop_answer_scores": [],
            "negative_crop_answer_scores": [],
            "masked_crop_answer_scores": [],
            "evidence_gap": 0.0,
            "support_gap": 0.0,
            "mask_drop": 0.0,
            "crop_verified": False,
            "selected_positive_views": [],
            "selected_negative_views": [],
        }

    def _score_candidate_view_support(self, answer_text, clues, media=None, problem=None):
        clue_info = self._empty_clue_info()
        if clues:
            clue_info = self._score_clues(
                answer_text=answer_text,
                clues=clues,
                media=media,
                problem=problem,
            )

        answer_info = self._score_answer_only(
            question=None,
            answer_text=answer_text,
            media=media,
            problem=problem,
        )

        if clues:
            selection_score = clue_info["clue_score"] + 0.25 * answer_info["answer_score"]
        else:
            selection_score = answer_info["answer_score"]

        return {
            "selection_score": float(selection_score),
            "clue_info": clue_info,
            "answer_info": answer_info,
        }

    def _select_candidate_views_for_answer(self, answer_text, clues, crop_evidence, problem=None):
        candidate_views = crop_evidence.get("candidate_views") or []
        if not candidate_views:
            positive_views = crop_evidence.get("positive_views") or crop_evidence.get("proposal_positive_views") or []
            negative_views = crop_evidence.get("negative_views") or crop_evidence.get("proposal_negative_views") or []
            return positive_views, negative_views

        ranked_views = []
        for view in candidate_views:
            support = self._score_candidate_view_support(
                answer_text=answer_text,
                clues=clues,
                media=view.get("media"),
                problem=problem,
            )
            ranked_view = dict(view)
            ranked_view["support_score"] = support["selection_score"]
            ranked_view["support_clue_reward"] = support["clue_info"]["clue_reward"]
            ranked_view["support_answer_reward"] = support["answer_info"]["answer_reward"]
            ranked_views.append(ranked_view)

        ranked_views.sort(key=lambda item: item.get("support_score", float("-inf")), reverse=True)
        top_k_positive = max(int(crop_evidence.get("top_k_positive", 2)), 0)
        max_negative = max(int(crop_evidence.get("max_negative", 3)), 0)

        positive_views = ranked_views[:top_k_positive] if top_k_positive > 0 else []
        positive_ids = {view.get("view_id") for view in positive_views}
        negative_candidates = [view for view in ranked_views if view.get("view_id") not in positive_ids]
        negative_views = negative_candidates[-max_negative:] if max_negative > 0 else []
        return positive_views, negative_views

    def _score_crop_evidence(self, answer_text, crop_evidence, media=None, problem=None):
        empty = self._empty_crop_info()
        if self.visual_reward is None or self.crop_weight <= 0 or not crop_evidence:
            return empty

        positive_views, negative_views = self._select_candidate_views_for_answer(
            answer_text=answer_text,
            clues=crop_evidence.get("clues"),
            crop_evidence=crop_evidence,
            problem=problem,
        )
        if not positive_views and not negative_views:
            return empty

        full_info = self._score_answer_only(
            question=None,
            answer_text=answer_text,
            media=media,
            problem=problem,
        )
        full_answer_score = float(full_info["answer_score"])

        positive_scores = []
        positive_answer_scores = []
        for view in positive_views:
            info = self._score_answer_only(
                question=None,
                answer_text=answer_text,
                media=view.get("media"),
                problem=problem,
            )
            positive_answer_scores.append(float(info["answer_score"]))
            positive_scores.append(float(info["answer_reward"]))

        negative_scores = []
        negative_answer_scores = []
        for view in negative_views:
            info = self._score_answer_only(
                question=None,
                answer_text=answer_text,
                media=view.get("media"),
                problem=problem,
            )
            negative_answer_scores.append(float(info["answer_score"]))
            negative_scores.append(float(info["answer_reward"]))

        masked_scores = []
        masked_answer_scores = []
        primary_image = get_primary_image(media)
        if primary_image is not None:
            for view in positive_views:
                masked_image = build_masked_view(primary_image, view.get("bbox"))
                if masked_image is None:
                    continue
                masked_media = build_view_media(masked_image)
                info = self._score_answer_only(
                    question=None,
                    answer_text=answer_text,
                    media=masked_media,
                    problem=problem,
                )
                masked_answer_scores.append(float(info["answer_score"]))
                masked_scores.append(float(info["answer_reward"]))

        positive_term = (
            sum(positive_answer_scores) / len(positive_answer_scores)
            if positive_answer_scores
            else full_answer_score
        )
        negative_term = (
            sum(negative_answer_scores) / len(negative_answer_scores)
            if negative_answer_scores
            else full_answer_score
        )
        masked_term = (
            sum(masked_answer_scores) / len(masked_answer_scores)
            if masked_answer_scores
            else full_answer_score
        )

        support_gap = positive_term - negative_term if positive_answer_scores and negative_answer_scores else 0.0
        mask_drop = full_answer_score - masked_term if masked_answer_scores else 0.0
        evidence_terms = []
        if positive_answer_scores and negative_answer_scores:
            evidence_terms.append(support_gap)
        if masked_answer_scores:
            evidence_terms.append(mask_drop)
        evidence_gap = sum(evidence_terms) / len(evidence_terms) if evidence_terms else 0.0
        crop_reward = self._squash_delta(evidence_gap)

        return {
            "crop_evidence_score": float(evidence_gap),
            "crop_reward": float(crop_reward),
            "positive_crop_scores": [float(item) for item in positive_scores],
            "negative_crop_scores": [float(item) for item in negative_scores],
            "masked_crop_scores": [float(item) for item in masked_scores],
            "positive_crop_answer_scores": [float(item) for item in positive_answer_scores],
            "negative_crop_answer_scores": [float(item) for item in negative_answer_scores],
            "masked_crop_answer_scores": [float(item) for item in masked_answer_scores],
            "evidence_gap": float(evidence_gap),
            "support_gap": float(support_gap),
            "mask_drop": float(mask_drop),
            "crop_verified": True,
            "selected_positive_views": summarize_views(positive_views),
            "selected_negative_views": summarize_views(negative_views),
        }

    def score_candidates(
        self,
        question,
        answer_texts,
        media=None,
        problem=None,
        clues=None,
        crop_evidence=None,
        reference_answer=None,
    ):
        if answer_texts is None:
            return []

        media = media or {}
        answer_texts = list(answer_texts)
        if not answer_texts:
            return []

        def answer_key(text):
            return self._normalize_answer_display(text, problem=problem)

        score_map = {}
        ordered_texts = list(answer_texts)
        if reference_answer is not None:
            ordered_texts.append(reference_answer)

        for text in ordered_texts:
            key = answer_key(text)
            if key in score_map:
                continue
            score_map[key] = self._score_answer_only(
                question=question,
                answer_text=key,
                media=media,
                problem=problem,
            )

        clue_map = {}
        if clues:
            for text in ordered_texts:
                key = answer_key(text)
                if key in clue_map:
                    continue
                clue_map[key] = self._score_clues(
                    answer_text=key,
                    clues=clues,
                    media=media,
                    problem=problem,
                )

        if crop_evidence is not None:
            crop_evidence = dict(crop_evidence)
            crop_evidence["clues"] = list(clues or [])

        crop_map = {}
        if crop_evidence and self.crop_weight > 0:
            # Crop verification is keyed by normalized answer text, so select crop
            # candidates on unique answers and score each unique answer once.
            preliminary_by_key = {}
            key_order = []
            for idx, text in enumerate(answer_texts):
                key = answer_key(text)
                if key not in preliminary_by_key:
                    key_order.append(key)
                answer_score = score_map[key]["answer_score"]
                clue_score = clue_map.get(key, {}).get("clue_score", 0.0)
                preliminary_score = self.answer_weight * answer_score + 0.25 * clue_score
                previous = preliminary_by_key.get(key)
                if previous is None or preliminary_score > previous["score"]:
                    preliminary_by_key[key] = {
                        "score": float(preliminary_score),
                    }
            preliminary_scores = [
                (
                    key,
                    info["score"],
                )
                for key, info in preliminary_by_key.items()
            ]
            preliminary_scores.sort(key=lambda item: item[1], reverse=True)
            if self.max_crop_candidates_per_step <= 0:
                max_crop_candidates = max(len(preliminary_scores), 1)
            else:
                max_crop_candidates = self.max_crop_candidates_per_step
            selected_crop_keys = {
                key
                for key, _ in preliminary_scores[:max_crop_candidates]
            }
            for key in key_order:
                if key not in selected_crop_keys:
                    crop_map[key] = self._empty_crop_info()
                    continue
                crop_map[key] = self._score_crop_evidence(
                    answer_text=key,
                    crop_evidence=crop_evidence,
                    media=media,
                    problem=problem,
                )

        reference_reward = None
        if reference_answer is not None:
            reference_reward = score_map[answer_key(reference_answer)]["answer_reward"]
        reference_answer_score = None
        if reference_answer is not None:
            reference_answer_score = score_map[answer_key(reference_answer)]["answer_score"]

        reference_clue_score = None
        if reference_answer is not None and clue_map:
            reference_clue_score = clue_map[answer_key(reference_answer)]["clue_score"]

        candidate_answer_scores = [
            score_map[answer_key(text)]["answer_score"]
            for text in answer_texts
        ]
        candidate_answer_rewards = [
            score_map[answer_key(text)]["answer_reward"]
            for text in answer_texts
        ]
        candidate_clue_scores = [
            clue_map.get(answer_key(text), {}).get("clue_score", 0.0)
            for text in answer_texts
        ]

        results = []
        for idx, text in enumerate(answer_texts):
            key = answer_key(text)
            base_info = dict(score_map[key])
            clue_info = clue_map.get(key, self._empty_clue_info())
            crop_info = crop_map.get(key, self._empty_crop_info())

            competitors = []
            competitor_scores = []
            if reference_reward is not None:
                competitors.append(reference_reward)
            if reference_answer_score is not None:
                competitor_scores.append(reference_answer_score)
            for jdx, sibling_reward in enumerate(candidate_answer_rewards):
                if jdx == idx:
                    continue
                competitors.append(sibling_reward)
            for jdx, sibling_answer_score in enumerate(candidate_answer_scores):
                if jdx == idx:
                    continue
                competitor_scores.append(sibling_answer_score)

            competitor_best_reward = max(competitors) if competitors else base_info["answer_reward"]
            competitor_best_answer_score = max(competitor_scores) if competitor_scores else base_info["answer_score"]
            answer_margin_delta = base_info["answer_score"] - competitor_best_answer_score
            margin_score = self._squash_delta(answer_margin_delta)

            clue_competitors = []
            if reference_clue_score is not None:
                clue_competitors.append(reference_clue_score)
            for jdx, sibling_clue_score in enumerate(candidate_clue_scores):
                if jdx == idx:
                    continue
                clue_competitors.append(sibling_clue_score)
            competitor_clue_baseline = (
                sum(clue_competitors) / len(clue_competitors)
                if clue_competitors
                else clue_info["clue_score"]
            )
            clue_margin_delta = clue_info["clue_score"] - competitor_clue_baseline
            clue_margin_score = self._squash_delta(clue_margin_delta)

            total_score = (
                self.answer_weight * base_info["answer_reward"]
                + self.clue_weight * clue_margin_score
                + self.crop_weight * crop_info["crop_reward"]
                + self.margin_weight * margin_score
            )

            base_info.update(
                {
                    "clue_score": float(clue_info["clue_score"]),
                    "clue_reward": float(clue_info["clue_reward"]),
                    "clue_visual_logprob": clue_info["clue_visual_logprob"],
                    "clue_prior_logprob": clue_info["clue_prior_logprob"],
                    "clue_text": clue_info["clue_text"],
                    "clue_margin_delta": float(clue_margin_delta),
                    "clue_margin_score": float(clue_margin_score),
                    "competitor_clue_baseline": float(competitor_clue_baseline),
                    "crop_evidence_score": float(crop_info["crop_evidence_score"]),
                    "crop_reward": float(crop_info["crop_reward"]),
                    "positive_crop_scores": list(crop_info["positive_crop_scores"]),
                    "negative_crop_scores": list(crop_info["negative_crop_scores"]),
                    "masked_crop_scores": list(crop_info["masked_crop_scores"]),
                    "positive_crop_answer_scores": list(crop_info["positive_crop_answer_scores"]),
                    "negative_crop_answer_scores": list(crop_info["negative_crop_answer_scores"]),
                    "masked_crop_answer_scores": list(crop_info["masked_crop_answer_scores"]),
                    "evidence_gap": float(crop_info["evidence_gap"]),
                    "support_gap": float(crop_info["support_gap"]),
                    "mask_drop": float(crop_info["mask_drop"]),
                    "crop_verified": bool(crop_info["crop_verified"]),
                    "selected_positive_views": list(crop_info["selected_positive_views"]),
                    "selected_negative_views": list(crop_info["selected_negative_views"]),
                    "answer_margin_delta": float(answer_margin_delta),
                    "margin_score": float(margin_score),
                    "competitor_best_answer_score": float(competitor_best_answer_score),
                    "competitor_best_reward": float(competitor_best_reward),
                    "reference_answer_score": None if reference_answer_score is None else float(reference_answer_score),
                    "reference_reward": None if reference_reward is None else float(reference_reward),
                    "total_score": float(total_score),
                }
            )
            results.append(base_info)

        return results
