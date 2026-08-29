from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, fields
from typing import Literal

Strength = Literal["hard", "soft"]
Source = Literal["current_turn", "conversation", "profile"]


@dataclass(frozen=True)
class Constraint:
    value: str | float
    strength: Strength
    source: Source
    updated_turn: int


@dataclass(frozen=True)
class PriceConstraint:
    minimum: float | None = None
    maximum: float | None = None
    strength: Strength = "hard"
    source: Source = "current_turn"
    updated_turn: int = 0


@dataclass(frozen=True)
class SearchQuery:
    text: str
    category: Constraint | None = None
    color: Constraint | None = None
    style: Constraint | None = None
    material: Constraint | None = None
    use_case: Constraint | None = None
    price: PriceConstraint | None = None
    exclusions: dict[str, set[str]] | None = None


@dataclass
class SessionState:
    category: Constraint | None = None
    color: Constraint | None = None
    style: Constraint | None = None
    material: Constraint | None = None
    use_case: Constraint | None = None
    price: PriceConstraint | None = None
    exclusions: dict[str, set[str]] = field(default_factory=dict)
    removed_constraints: set[str] = field(default_factory=set)
    raw_current_turn_text: str = ""


SLOT_NAMES = ("category", "color", "style", "material", "use_case")

# Aliases are intentionally finite. The extractor only emits values explicitly
# supported by the message and never guesses a slot with an LLM.
SLOT_ALIASES: dict[str, tuple[tuple[str, str], ...]] = {
    "category": (
        ("running shoes", "sneakers"),
        ("running shoe", "sneakers"),
        ("tennis shoes", "sneakers"),
        ("tennis shoe", "sneakers"),
        ("sneakers", "sneakers"),
        ("sneaker", "sneakers"),
        ("trainers", "sneakers"),
        ("trainer", "sneakers"),
        ("t-shirts", "t-shirt"),
        ("t-shirt", "t-shirt"),
        ("tee shirts", "t-shirt"),
        ("tee shirt", "t-shirt"),
        ("handbags", "handbag"),
        ("handbag", "handbag"),
        ("backpacks", "backpack"),
        ("backpack", "backpack"),
        ("sunglasses", "sunglasses"),
        ("earrings", "earrings"),
        ("necklaces", "necklace"),
        ("necklace", "necklace"),
        ("bracelets", "bracelet"),
        ("bracelet", "bracelet"),
        ("dresses", "dress"),
        ("dress", "dress"),
        ("jackets", "jacket"),
        ("jacket", "jacket"),
        ("sweaters", "sweater"),
        ("sweater", "sweater"),
        ("hoodies", "hoodie"),
        ("hoodie", "hoodie"),
        ("sandals", "sandals"),
        ("boots", "boots"),
        ("boot", "boots"),
        ("shoes", "shoes"),
        ("shoe", "shoes"),
        ("shirts", "shirt"),
        ("shirt", "shirt"),
        ("blouses", "blouse"),
        ("blouse", "blouse"),
        ("coats", "coat"),
        ("coat", "coat"),
        ("jeans", "jeans"),
        ("pants", "pants"),
        ("trousers", "pants"),
        ("shorts", "shorts"),
        ("skirts", "skirt"),
        ("skirt", "skirt"),
        ("socks", "socks"),
        ("hats", "hat"),
        ("hat", "hat"),
        ("bags", "bag"),
        ("bag", "bag"),
        ("watches", "watch"),
        ("watch", "watch"),
        ("rings", "ring"),
        ("ring", "ring"),
    ),
    "color": (
        ("navy blue", "navy"),
        ("light blue", "light blue"),
        ("dark blue", "dark blue"),
        ("rose gold", "rose gold"),
        ("black", "black"),
        ("white", "white"),
        ("blue", "blue"),
        ("red", "red"),
        ("pink", "pink"),
        ("green", "green"),
        ("brown", "brown"),
        ("grey", "gray"),
        ("gray", "gray"),
        ("purple", "purple"),
        ("yellow", "yellow"),
        ("orange", "orange"),
        ("beige", "beige"),
        ("navy", "navy"),
        ("gold", "gold"),
        ("silver", "silver"),
        ("tan", "tan"),
    ),
    "style": (
        ("business casual", "business casual"),
        ("smart casual", "smart casual"),
        ("minimalist", "minimalist"),
        ("vintage", "vintage"),
        ("classic", "classic"),
        ("elegant", "elegant"),
        ("bohemian", "bohemian"),
        ("casual", "casual"),
        ("formal", "formal"),
        ("sporty", "sporty"),
        ("modern", "modern"),
        ("slim fit", "slim fit"),
        ("relaxed fit", "relaxed fit"),
        ("oversized", "oversized"),
    ),
    "material": (
        ("faux leather", "faux leather"),
        ("genuine leather", "leather"),
        ("stainless steel", "stainless steel"),
        ("sterling silver", "sterling silver"),
        ("cotton", "cotton"),
        ("polyester", "polyester"),
        ("nylon", "nylon"),
        ("leather", "leather"),
        ("canvas", "canvas"),
        ("wool", "wool"),
        ("spandex", "spandex"),
        ("silk", "silk"),
        ("rayon", "rayon"),
        ("linen", "linen"),
        ("denim", "denim"),
        ("suede", "suede"),
        ("rubber", "rubber"),
        ("fabric", "fabric"),
    ),
    "use_case": (
        ("everyday wear", "everyday"),
        ("daily wear", "everyday"),
        ("working out", "workout"),
        ("work outs", "workout"),
        ("workout", "workout"),
        ("running", "running"),
        ("jogging", "running"),
        ("hiking", "hiking"),
        ("walking", "walking"),
        ("traveling", "travel"),
        ("travelling", "travel"),
        ("travel", "travel"),
        ("wedding", "wedding"),
        ("office", "office"),
        ("for work", "work"),
        ("gym", "gym"),
        ("winter", "winter"),
        ("outdoor", "outdoor"),
        ("sports", "sports"),
        ("everyday", "everyday"),
    ),
}

PRICE_RANGE_RE = re.compile(
    r"(?:between|from)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_MAX_RE = re.compile(
    r"(?:under|below|less than|no more than|at most|up to|max(?:imum)?(?: of)?)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_MIN_RE = re.compile(
    r"(?:over|above|more than|at least|min(?:imum)?(?: of)?)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_BUDGET_RE = re.compile(
    r"(?:budget(?: is| of| around)?|around)\s*\$\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
PRICE_REMOVAL_RE = re.compile(
    r"(?:budget|price|cost)(?:\s+(?:range|limit))?\s+(?:is\s+)?(?:no longer important|doesn['’]?t matter|isn['’]?t important)|"
    r"(?:no|without)\s+(?:a\s+)?(?:budget|price|cost)(?:\s+(?:limit|constraint))?",
    re.IGNORECASE,
)
SOFT_CUE_RE = re.compile(r"\b(?:prefer|preference|ideally|if possible|would like)\b", re.IGNORECASE)
OVERRIDE_CUE_RE = re.compile(r"\b(?:actually|instead|rather|changed my mind|ignore my earlier)\b", re.IGNORECASE)
GENERIC_REMOVAL_RE = re.compile(
    r"(?:don't|do not)\s+(?:need|want|care about)\s+([a-z][a-z0-9 -]{0,40}?)\s+anymore\b|"
    r"no longer\s+(?:need|want)\s+([a-z][a-z0-9 -]{0,40}?)(?:[.;,]|$)",
    re.IGNORECASE,
)


def _normalise_message(message: str) -> str:
    return re.sub(r"\s+", " ", message.lower().replace("’", "'").replace("—", " ").replace("–", " ")).strip()


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", re.IGNORECASE)


def _find_slot_mentions(text: str, slot: str) -> list[tuple[int, int, str]]:
    mentions: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for alias, canonical in SLOT_ALIASES[slot]:
        for match in _alias_pattern(alias).finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            mentions.append((span[0], span[1], canonical))
            occupied.append(span)
    return sorted(mentions)


def _is_directly_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 32):start]
    return bool(re.search(r"(?:\bnot|\bno|\bwithout|\bexcept|\bavoid|\bdon't want|\bdo not want)\s+(?:any\s+)?$", prefix))


def _is_removed_value(text: str, start: int, end: int) -> bool:
    prefix = text[max(0, start - 32):start]
    suffix = text[end:min(len(text), end + 24)]
    if re.search(r"no longer (?:need|want|like)\s+$", prefix):
        return True
    return bool(
        re.search(r"(?:don't|do not) (?:need|want)\s+$", prefix)
        and re.search(r"\banymore\b", suffix)
    )


def _constraint(value: str, source: Source, turn: int, strength: Strength | None = None) -> Constraint:
    resolved_strength: Strength = strength or ("soft" if source == "profile" else "hard")
    return Constraint(value=value, strength=resolved_strength, source=source, updated_turn=turn)


def _price_constraint(
    minimum: float | None,
    maximum: float | None,
    source: Source,
    turn: int,
    strength: Strength | None = None,
) -> PriceConstraint:
    resolved_strength: Strength = strength or ("soft" if source == "profile" else "hard")
    return PriceConstraint(
        minimum=minimum,
        maximum=maximum,
        strength=resolved_strength,
        source=source,
        updated_turn=turn,
    )


def _extract_price(text: str, source: Source, turn: int, strength: Strength | None) -> PriceConstraint | None:
    range_match = PRICE_RANGE_RE.search(text)
    if range_match:
        first, second = (float(range_match.group(1)), float(range_match.group(2)))
        return _price_constraint(min(first, second), max(first, second), source, turn, strength)
    maximum_match = PRICE_MAX_RE.search(text)
    minimum_match = PRICE_MIN_RE.search(text)
    if maximum_match or minimum_match:
        return _price_constraint(
            float(minimum_match.group(1)) if minimum_match else None,
            float(maximum_match.group(1)) if maximum_match else None,
            source,
            turn,
            strength,
        )
    budget_match = PRICE_BUDGET_RE.search(text)
    if budget_match:
        return _price_constraint(None, float(budget_match.group(1)), source, turn, strength)
    return None


def _profile_text(user_profile: dict) -> str:
    parts: list[str] = []
    tags = user_profile.get("preference_tags")
    if isinstance(tags, list):
        parts.extend(str(tag) for tag in tags)
    summary = user_profile.get("summary")
    if isinstance(summary, str):
        parts.append(summary)
    return " ".join(parts)


def _demote_previous_turn(state: SessionState) -> None:
    for slot in SLOT_NAMES:
        value = getattr(state, slot)
        if value is not None and value.source == "current_turn":
            setattr(
                state,
                slot,
                Constraint(value.value, value.strength, "conversation", value.updated_turn),
            )
    if state.price is not None and state.price.source == "current_turn":
        state.price = PriceConstraint(
            minimum=state.price.minimum,
            maximum=state.price.maximum,
            strength=state.price.strength,
            source="conversation",
            updated_turn=state.price.updated_turn,
        )


def _clear_slot(state: SessionState, slot: str) -> None:
    value = getattr(state, slot)
    if value is not None:
        state.removed_constraints.add(f"{slot}:{value.value}")
    setattr(state, slot, None)


def _remove_named_slots(state: SessionState, text: str) -> None:
    slot_phrases = {
        "category": "category",
        "color": "color",
        "colour": "color",
        "style": "style",
        "material": "material",
        "use case": "use_case",
        "occasion": "use_case",
    }
    for phrase, slot in slot_phrases.items():
        escaped = re.escape(phrase)
        if re.search(
            rf"(?:no|without)\s+(?:a\s+)?{escaped}\s+(?:preference|requirement)|"
            rf"{escaped}\s+(?:doesn't|does not|isn't|is not|no longer)\s+(?:matter|important|required)",
            text,
        ):
            _clear_slot(state, slot)
            state.removed_constraints.add(slot)


def _record_unsupported_removals(
    state: SessionState,
    text: str,
    removed_supported_values: set[str],
) -> None:
    for match in GENERIC_REMOVAL_RE.finditer(text):
        value = next(group for group in match.groups() if group is not None)
        value = re.sub(r"\s+", " ", value).strip(" -")
        if value and value not in removed_supported_values:
            state.removed_constraints.add(f"feature:{value}")


def _apply_profile(state: SessionState, user_profile: dict) -> None:
    text = _normalise_message(_profile_text(user_profile))
    if not text:
        return
    for slot in SLOT_NAMES:
        mentions = _find_slot_mentions(text, slot)
        if mentions:
            setattr(state, slot, _constraint(mentions[-1][2], "profile", 0))
    state.price = _extract_price(text, "profile", 0, "soft")


def _format_amount(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:g}"


def build_search_query(state: SessionState) -> SearchQuery:
    terms: list[str] = []
    for slot in ("category", "style", "use_case", "color", "material"):
        value = getattr(state, slot)
        if value is not None:
            rendered = str(value.value)
            if rendered not in terms:
                terms.append(rendered)
    if state.price is not None:
        if state.price.minimum is not None and state.price.maximum is not None:
            terms.append(
                f"${_format_amount(state.price.minimum)} to ${_format_amount(state.price.maximum)}"
            )
        elif state.price.maximum is not None:
            terms.append(f"under ${_format_amount(state.price.maximum)}")
        elif state.price.minimum is not None:
            terms.append(f"over ${_format_amount(state.price.minimum)}")
    exclusions = {slot: set(values) for slot, values in state.exclusions.items() if values}
    return SearchQuery(
        text=" ".join(terms),
        category=state.category,
        color=state.color,
        style=state.style,
        material=state.material,
        use_case=state.use_case,
        price=state.price,
        exclusions=exclusions or None,
    )


class ConversationStateManager:
    """Owns deterministic, isolated conversation state for many sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict | None = None) -> SessionState:
        if not session_id:
            raise ValueError("session_id must not be empty")
        state = SessionState()
        _apply_profile(state, user_profile or {})
        self._sessions[session_id] = state
        return copy.deepcopy(state)

    def state_for(self, session_id: str) -> SessionState:
        return copy.deepcopy(self._require_session(session_id))

    def query_for(self, session_id: str) -> SearchQuery:
        return build_search_query(self._require_session(session_id))

    def update(self, session_id: str, user_message: str, turn: int) -> SearchQuery:
        if turn < 1:
            raise ValueError("turn must be at least 1")
        state = self._require_session(session_id)
        state.raw_current_turn_text = user_message
        text = _normalise_message(user_message)
        _demote_previous_turn(state)
        _remove_named_slots(state, text)

        if PRICE_REMOVAL_RE.search(text):
            if state.price is not None:
                state.removed_constraints.add("price")
            state.price = None

        mentions_by_slot = {slot: _find_slot_mentions(text, slot) for slot in SLOT_NAMES}
        positive_by_slot: dict[str, str] = {}
        exclusions_by_slot: dict[str, set[str]] = {}
        removed_supported_values: set[str] = set()

        for slot, mentions in mentions_by_slot.items():
            for start, end, value in mentions:
                if _is_removed_value(text, start, end):
                    removed_supported_values.add(value)
                    current = getattr(state, slot)
                    if current is not None and current.value == value:
                        _clear_slot(state, slot)
                    state.removed_constraints.add(f"{slot}:{value}")
                elif _is_directly_negated(text, start):
                    exclusions_by_slot.setdefault(slot, set()).add(value)
                else:
                    positive_by_slot[slot] = value

        _record_unsupported_removals(state, text, removed_supported_values)

        override = bool(OVERRIDE_CUE_RE.search(text))
        new_category = positive_by_slot.get("category")
        old_category = state.category.value if state.category is not None else None
        if new_category is not None and (new_category != old_category or override):
            _clear_slot(state, "style")
            _clear_slot(state, "use_case")

        new_use_case = positive_by_slot.get("use_case")
        old_use_case = state.use_case.value if state.use_case is not None else None
        if new_use_case is not None and new_use_case != old_use_case:
            _clear_slot(state, "use_case")
            if override:
                _clear_slot(state, "style")
        elif override and positive_by_slot.get("style") is not None:
            _clear_slot(state, "use_case")

        for slot, values in exclusions_by_slot.items():
            state.exclusions.setdefault(slot, set()).update(values)
            current = getattr(state, slot)
            if current is not None and str(current.value) in values:
                _clear_slot(state, slot)

        strength: Strength = "soft" if SOFT_CUE_RE.search(text) else "hard"
        for slot, value in positive_by_slot.items():
            state.exclusions.setdefault(slot, set()).discard(value)
            if not state.exclusions[slot]:
                state.exclusions.pop(slot)
            previous = getattr(state, slot)
            if previous is not None and previous.value != value:
                state.removed_constraints.add(f"{slot}:{previous.value}")
            setattr(state, slot, _constraint(value, "current_turn", turn, strength))
            state.removed_constraints.discard(f"{slot}:{value}")
            state.removed_constraints.discard(slot)

        if not PRICE_REMOVAL_RE.search(text):
            price = _extract_price(text, "current_turn", turn, strength)
            if price is not None:
                state.price = price
                state.removed_constraints.discard("price")

        return build_search_query(state)

    def _require_session(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise RuntimeError("reset must be called before updating or querying a session") from error


def slot_dict(state: SessionState) -> dict[str, Constraint | PriceConstraint | None]:
    """Return the active slot fields without mutable bookkeeping collections."""
    return {item.name: getattr(state, item.name) for item in fields(state) if item.name in (*SLOT_NAMES, "price")}
