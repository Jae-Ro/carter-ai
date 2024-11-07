from llama_models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)
from llama_models.llama3.reference_impl.generation import Llama
from typing import Optional, List, Union

import os, argparse, re

PROMPT = """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]

Constraints:
* 2 <= nums.length <= 104
* 109 <= nums[i] <= 109
* 109 <= target <= 109
* Only one valid answer exists."""


SYSTEM_GUIDANCE = """Follow these rules when giving responses:
Imagine you're an interviewer conducting a live Data Structures and Algorithms interview with a candidate and can only give short responses.
Be positive and encouraging to the candidate.
Do NOT provide any python code or solutions to the user until they clearly state they have given up.
Do not ask the user what problem they're trying to solve.
Don't ask the user to consider any improvements or optimizations of their approach until after they have implemented their solution in code.
If the user is unsure about your feedback, try and explain the previous response more simply.
If the user is asking for hints, first look at evaluating their approach and see if it generally solves the problem (even if it's not an optimal solution).
If the user is able to implement a solution that compiles and solves the problem, then ask how they might improve their solution based on Time and Space complexity.
Keep this back and forth of question and answering with hints going until the user has either achieved at least a semi-optimal solution or gives up and asks for the answer.
"""

def is_valid_args(args: dict) -> bool:
    checkpoint_path = args['checkpoint']
    verbose = args['verbose']

    correct_checkpoint = False
    correct_verbose = False

    if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
        correct_checkpoint = True
    
    if isinstance(verbose, bool): correct_verbose = True

    if all([correct_checkpoint, correct_verbose]): return True
    return False


def contains_python_function(input_string):
    # Define the regular expression pattern for a Python function definition
    pattern = r'\bdef\s+\w+\s*\(.*\)\s*:'

    # Search for the pattern in the input string
    return bool(re.search(pattern, input_string))


def normalize_whitespace(input_string):
    # Replace all sequences of whitespace characters (spaces, newlines, tabs, etc.) with a single space
    normalized_string = re.sub(r'\s+', ' ', input_string)
    
    # Optionally strip leading/trailing whitespace from the result
    return normalized_string.strip()


def get_model(
    ckpt_dir: str,
    max_seq_len: int = 5000,
    max_batch_size: int = 4,
    model_parallel_size: Optional[int] = None,
) -> Llama:
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        model_parallel_size=model_parallel_size,
    )
    return generator


def simple_question_answer(
    generator: Llama,
    dialog: List[Union[UserMessage, CompletionMessage, SystemMessage, StopReason]],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: Optional[int] = None,
) -> any:
    """_summary_

    Args:
        generator (Llama): _description_
        dialog (List[Union[UserMessage, CompletionMessage, SystemMessage, StopReason]]): _description_
        temperature (float, optional): _description_. Defaults to 0.6.
        top_p (float, optional): _description_. Defaults to 0.9.
        max_gen_len (Optional[int], optional): _description_. Defaults to None.

    Returns:
        any: _description_
    """
    result = generator.chat_completion(
        dialog,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for msg in dialog:
        if isinstance(msg, SystemMessage): continue
        print(f"\n{msg.role.capitalize()}: {msg.content}\n")

    out_message = result.generation
    print(f"> {out_message.role.capitalize()}: {out_message.content}")
    print("\n==================================\n")
    
    return out_message
    

def summarize_conversation(
    conversation_history: List[str],
    min_length: int = 50,
    max_length: int = 300,
    generator = None
) -> str:
    conversation_text = "\n".join(conversation_history)
    # print(f"\n\nConversation Text:\n\n{conversation_text}\n\n")
    # conv_word_count = len(conversation_text.split("\n"))
    # if conv_word_count < max_length:
    #     max_length = 150
    result = generator.chat_completion(
        [UserMessage(content=f"Summarize each response from [USER] and [MODEL] briefly and maintain [USER] and\
            [MODEL] markings and do not include excess whitespace unless necessary: {conversation_text}")],
        max_gen_len=None,
        temperature=0.6,
        top_p=0.9,
    )
    out_message = result.generation
    summary_text = out_message.content

    return summary_text

def run(
    checkpoint_path: str, 
    verbose: bool = False,
    context: str = PROMPT,
    system_guidance: str = SYSTEM_GUIDANCE
):

    # Set required torch.distributed env variables
    os.environ['RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "29500"
    os.environ['USE_LIBUV'] = "1"

    system_guidance = normalize_whitespace(system_guidance)

    # get llm text generator -- call once per notebook run
    generator = get_model(checkpoint_path)
    
    fmt_prompt = normalize_whitespace(context)
    context_li = [f"[USER]: Help me solve this problem: {fmt_prompt}"]

    print(f"\nPROBLEM: {context}")
    initial_input = input(f"\nHow would you go about solving this?\n>>> ")
    while initial_input.strip() == "":
        initial_input = input("\nI'm sorry I didn't get that, could you please try again?\n>>> ")
    
    if "stop" in initial_input.lower() or "exit()" in initial_input.lower(): quit()

    dialog = [
        UserMessage(content=initial_input, context=summarize_conversation(context_li, generator=generator)), 
        SystemMessage(content=system_guidance)
    ]
    out_msg = simple_question_answer(generator, dialog)
    
    context_li.append(f"[USER]: {initial_input}")
    context_li.append(f"[MODEL]: {out_msg.content}")
    fmt_user, fmt_agent = initial_input, out_msg.content
    if not contains_python_function(fmt_user): fmt_user = normalize_whitespace(fmt_user)
    if not contains_python_function(fmt_agent): fmt_agent = normalize_whitespace(fmt_agent)

    context_li.append(f"[USER]: {fmt_user}")
    context_li.append(f"[MODEL]: {fmt_agent}")

    user_query = ""

    while True:
        user_query = input("\n>>> ")
        if user_query.strip() == "":
            print(f"\nI'm sorry I didn't get that, could you please try again?")
            continue
        
        if "stop" in user_query.lower() or "exit()" in user_query.lower(): quit()
        convo_summary = summarize_conversation(context_li, max_length=1000, generator=generator)
        if verbose: 
            # print(f"\nConext List: {context_li}")
            print(f"\nConversation History Summary:\n\t{convo_summary}")
        dialog = [UserMessage(content=user_query, context=convo_summary), SystemMessage(content=system_guidance)]
        out_msg = simple_question_answer(generator, dialog)

        # formatting for context
        fmt_user, fmt_agent = user_query, out_msg.content
        if not contains_python_function(fmt_user):
            fmt_user = normalize_whitespace(fmt_user)
        if not contains_python_function(fmt_agent):
            fmt_agent = normalize_whitespace(fmt_agent)

        context_li.append(f"[USER]: {fmt_user}")
        context_li.append(f"[MODEL]: {fmt_agent}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Carter AI",
        description="Example Pair Programming with LLama LLM",
        epilog="Try your best to solve the program with our AI assistant!" 
    )
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help="path to LLamam Model Checkpoint Directory (e.g. ~/.llama/checkpoints/Llama3.2-3B-Instruct)")
    parser.add_argument('-v', '--verbose', action="store_true", default=False, help="run in verbose mode for debuggin")
    args = parser.parse_args()
    args = vars(args)
    
    if is_valid_args(args):
        checkpoint_path = args['checkpoint']
        verbose = args['verbose']
        run(checkpoint_path, verbose=verbose)
    else:
        print(f"Invalid Arguments! Please try again.")