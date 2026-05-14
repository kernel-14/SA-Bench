## utils/helpers.py
import torch
from typing import List, Any, Tuple, Callable


def stop_gradient(tensor: torch.Tensor) -> torch.Tensor:
  """
  Detaches the given tensor from the current computational graph.
  This prevents gradients from flowing through this tensor.

  Args:
      tensor: The input torch.Tensor.

  Returns:
      A new torch.Tensor that is detached from the graph.
  """
  return tensor.detach()


def get_text_embeddings(
    prompts: List[str],
    text_encoder: Any,  # e.g., CLIPTextModel
    tokenizer: Any,  # e.g., CLIPTokenizer
    device: str,
    max_length: int,
) -> torch.Tensor:
  """
  Encodes a list of text prompts into embeddings using a given text encoder and tokenizer.

  Args:
      prompts: A list of strings, each representing a text prompt.
      text_encoder: An instantiated model capable of encoding text (e.g., CLIPTextModel).
                    It should already be loaded onto the specified device.
      tokenizer: An instantiated tokenizer corresponding to the text_encoder.
      device: The computational device ('cuda' or 'cpu').
      max_length: The maximum sequence length for tokenization, as per config.

  Returns:
      A torch.Tensor containing the text embeddings.
  """
  # Tokenize prompts
  text_inputs = tokenizer(
      prompts,
      padding="max_length",
      truncation=True,
      max_length=max_length,
      return_tensors="pt",
  )

  # Move tokenized inputs to the specified device
  input_ids = text_inputs.input_ids.to(device)
  attention_mask = text_inputs.attention_mask.to(device)

  # Generate embeddings using the text encoder
  text_encoder.eval()  # Set text encoder to evaluation mode
  with torch.no_grad():  # Disable gradient computation for the text encoder
    # For CLIPTextModel, typically last_hidden_state is used for cross-attention
    text_embeddings = text_encoder(
        input_ids=input_ids, attention_mask=attention_mask
    ).last_hidden_state

  return text_embeddings


def compute_jacobian_vector_product(
    func: Callable[[torch.Tensor], torch.Tensor],
    input_tensor: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
  """
  Computes the Jacobian-Vector Product (JVP) Jv, where J = ∂output/∂input.

  Args:
      func: The callable function f for which to compute JVP (output = f(input_tensor)).
            It should accept a single tensor argument and return a single tensor.
      input_tensor: The input tensor to the function f. It should have requires_grad=True.
      vector: A tensor representing the vector v for JVP, with the same shape as input_tensor.

  Returns:
      A torch.Tensor representing the Jacobian-Vector Product.
  """
  # jvp returns (output_val, jvp_result)
  _, jvp_result = torch.autograd.functional.jvp(func, (input_tensor,), (vector,))
  return jvp_result


def compute_vector_jacobian_product(
    output: torch.Tensor,
    input_tensor: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
  """
  Computes the Vector-Jacobian Product (VJP) v^T J, where J = ∂output/∂input.
  This is used in reverse-mode automatic differentiation.

  Args:
      output: The output tensor of the function f(input_tensor).
      input_tensor: The input tensor to the function f, with respect to which the Jacobian is taken.
                    It should have requires_grad=True or be part of a graph where it can be
                    differentiated.
      vector: A tensor representing the vector v (or grad_outputs) for VJP,
              with the same shape as output.

  Returns:
      A torch.Tensor representing the Vector-Jacobian Product.
      Returns the first element of the tuple returned by torch.autograd.grad.
  """
  # torch.autograd.grad returns a tuple of gradients for each input.
  # We assume input_tensor is a single tensor, so we take the first element.
  # retain_graph=True: Allows for multiple backward passes in the same graph if needed.
  # create_graph=False: We do not need to compute gradients of the VJP itself.
  vjp_result = torch.autograd.grad(
      outputs=output,
      inputs=input_tensor,
      grad_outputs=vector,
      retain_graph=True,
      create_graph=False,
  )[0]
  return vjp_result

