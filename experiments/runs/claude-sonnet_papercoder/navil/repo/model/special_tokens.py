## model/special_tokens.py
"""Special token definitions and tokenizer registration for NaViL.

This module defines the four structural special tokens used to encode
the spatial and multi-scale layout of visual content within the
multimodal token sequence. All components that handle image tokens
depend on this class for consistent token IDs.

Token insertion order (from the paper):
    [begin_of_image]
      [scale_0_row_0_tokens] [end_of_line]
      [scale_0_row_1_tokens] [end_of_line]
      ...
      [end_of_scale]
      [scale_1_row_0_tokens] [end_of_line]
      ...
      [end_of_scale]
    [end_of_image]

Modality mask convention: all four special tokens are treated as
visual tokens (modality=0) since they are structural markers within
the image token subsequence and route through the visual expert in
MoELayer.
"""

from typing import Dict

from transformers import PreTrainedTokenizer


class SpecialTokens:
    """Manages NaViL's four structural special tokens.

    This class defines the token strings, registers them with a
    HuggingFace tokenizer, and provides a single-entry-point accessor
    for their integer IDs. Token IDs are only valid after
    ``register_with_tokenizer`` has been called.

    Attributes:
        BEGIN_IMAGE: Token string marking the start of an image
            token subsequence.
        END_IMAGE: Token string marking the end of an image
            token subsequence.
        END_OF_LINE: Token string inserted after each row of image
            tokens to encode spatial row position.
        END_OF_SCALE: Token string inserted after each scale's tokens
            in multi-scale packing.
        token_ids: Mapping from token name (e.g. ``"BEGIN_IMAGE"``)
            to integer token ID. Empty until ``register_with_tokenizer``
            is called.

    Example::

        special_tokens = SpecialTokens()
        tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-1_8b")
        tokenizer = special_tokens.register_with_tokenizer(tokenizer)
        # Now resize model embeddings:
        model.resize_token_embeddings(len(tokenizer))
        begin_id = special_tokens.get_token_id("BEGIN_IMAGE")
    """

    # Class-level string constants — exact strings added to the tokenizer.
    # Other modules may reference these without instantiation, e.g.:
    #   SpecialTokens.BEGIN_IMAGE
    BEGIN_IMAGE: str = "<begin_of_image>"
    END_IMAGE: str = "<end_of_image>"
    END_OF_LINE: str = "<end_of_line>"
    END_OF_SCALE: str = "<end_of_scale>"

    # Ordered list of all token strings for iteration convenience.
    _ALL_TOKENS: tuple = (BEGIN_IMAGE, END_IMAGE, END_OF_LINE, END_OF_SCALE)

    # Mapping from token name string to its class-level constant string.
    # Used internally to validate ``get_token_id`` name arguments.
    _NAME_TO_STRING: Dict[str, str] = {
        "BEGIN_IMAGE": BEGIN_IMAGE,
        "END_IMAGE": END_IMAGE,
        "END_OF_LINE": END_OF_LINE,
        "END_OF_SCALE": END_OF_SCALE,
    }

    def __init__(self) -> None:
        """Initializes SpecialTokens with an empty token_ids mapping.

        Token IDs are populated only after ``register_with_tokenizer``
        is called. Attempting to call ``get_token_id`` before
        registration will raise a ``RuntimeError``.
        """
        self.token_ids: Dict[str, int] = {}

    def register_with_tokenizer(
        self, tokenizer: PreTrainedTokenizer
    ) -> PreTrainedTokenizer:
        """Adds the four special tokens to the tokenizer and records their IDs.

        This method must be called exactly once before any component
        attempts to use token IDs. After registration, the caller must
        resize the model's token embeddings to account for the newly
        added tokens::

            tokenizer = special_tokens.register_with_tokenizer(tokenizer)
            model.resize_token_embeddings(len(tokenizer))

        Args:
            tokenizer: A HuggingFace ``PreTrainedTokenizer`` instance
                to extend with the four NaViL special tokens.

        Returns:
            The same tokenizer instance, updated with the new special
            tokens. Returning it allows the caller to chain the call:
            ``tokenizer = special_tokens.register_with_tokenizer(tokenizer)``.

        Raises:
            ValueError: If any token ID resolves to the tokenizer's
                ``unk_token_id`` after registration, indicating that
                the token was not successfully added.
        """
        # Step 1: Add all four tokens as additional special tokens.
        # Using 'additional_special_tokens' (not bos/eos/etc.) because
        # these are custom structural tokens with no predefined HF role.
        num_added: int = tokenizer.add_special_tokens(
            {
                "additional_special_tokens": list(self._ALL_TOKENS)
            }
        )

        # Step 2: Retrieve integer IDs for each token.
        for name, token_string in self._NAME_TO_STRING.items():
            token_id: int = tokenizer.convert_tokens_to_ids(token_string)
            self.token_ids[name] = token_id

        # Step 3: Validate that no ID resolved to unk_token_id.
        # If a token was not registered successfully, convert_tokens_to_ids
        # returns unk_token_id, which would silently corrupt all downstream
        # modality masks and NTP loss computations.
        unk_id: int = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1
        invalid_tokens = [
            (name, token_id)
            for name, token_id in self.token_ids.items()
            if token_id == unk_id
        ]
        if invalid_tokens:
            invalid_names = [name for name, _ in invalid_tokens]
            raise ValueError(
                f"The following special tokens resolved to unk_token_id "
                f"({unk_id}) after registration, indicating they were not "
                f"successfully added to the tokenizer: {invalid_names}. "
                f"Ensure the tokenizer supports 'additional_special_tokens'."
            )

        # Log how many new tokens were actually added (0 if already present).
        # This is informational — re-registration is idempotent as long as
        # the same tokenizer instance is used.
        if num_added > 0:
            pass  # Tokens were newly added; embeddings must be resized.
        # If num_added == 0, tokens were already present; IDs are still valid.

        # Step 4: Return the updated tokenizer so the caller can chain:
        #   tokenizer = special_tokens.register_with_tokenizer(tokenizer)
        return tokenizer

    def get_token_id(self, name: str) -> int:
        """Returns the integer token ID for a given token name.

        Args:
            name: The token name string, one of ``"BEGIN_IMAGE"``,
                ``"END_IMAGE"``, ``"END_OF_LINE"``, or
                ``"END_OF_SCALE"``.

        Returns:
            The integer token ID as registered in the tokenizer.

        Raises:
            RuntimeError: If ``register_with_tokenizer`` has not been
                called yet (``token_ids`` is empty).
            KeyError: If ``name`` is not one of the four valid token
                name strings.

        Example::

            begin_id = special_tokens.get_token_id("BEGIN_IMAGE")
            end_id   = special_tokens.get_token_id("END_IMAGE")
            eol_id   = special_tokens.get_token_id("END_OF_LINE")
            eos_id   = special_tokens.get_token_id("END_OF_SCALE")
        """
        if not self.token_ids:
            raise RuntimeError(
                "SpecialTokens.get_token_id() called before "
                "register_with_tokenizer(). Call register_with_tokenizer(tokenizer) "
                "first to populate token IDs."
            )

        if name not in self._NAME_TO_STRING:
            valid_names = list(self._NAME_TO_STRING.keys())
            raise KeyError(
                f"Unknown token name '{name}'. "
                f"Valid token names are: {valid_names}."
            )

        return self.token_ids[name]
