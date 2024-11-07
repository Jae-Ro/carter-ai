from llama_models.llama3.api.datatypes import (
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)
from llama_models.llama3.reference_impl.generation import Llama
from transformers import pipeline

from typing import Optional, List, Union
import os, argparse


def is_valid_args(args: dict) -> bool:
    checkpoint_path = args['checkpoint']
    if os.path.exists(checkpoint_path) and os.path.isdir(checkpoint_path):
        return True

    return False


def run(checkpoint_path: str):

    def get_model(
        ckpt_dir: str,
        max_seq_len: int = 512,
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


    # Set required torch.distributed env variables
    os.environ['RANK'] = "0"
    os.environ['WORLD_SIZE'] = "1"
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "29500"
    os.environ['USE_LIBUV'] = "1"

    # get llm text generator -- call once per notebook run
    generator = get_model(checkpoint_path)


    summarizer = pipeline("summarization", model = "facebook/bart-large-cnn", device="cuda:0")
    print(summarizer)


    def summarize_conversation(
        conversation_history: List[str],
        min_length: int = 50,
        max_length: int = 300
    ) -> str:
        conversation_text = "\n".join(conversation_history)
        conv_word_count = len(conversation_text.split("\n"))
        if conv_word_count < max_length:
            max_length = 150
        summary = summarizer(conversation_text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']


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


    # initial prompt
    context = """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
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


    system_guidance = """Follow these rules when giving responses.
    Imagine you're a technical interviewer and can only give short responses.
    No matter what, don't provide any code to the user until they've said they've given up and want the answer. 
    Don't ask the user to consider any improvements or optimizations of their approach until after they have implemented their solution in code.
    If the user is unsure about your feedback, try and explain the previous response more simply.
    If the user is asking for hints, first look at evaluating their approach and see if it generally solves the problem (even if it's not an optimal solution).
    If the user is able to implement a solution that compiles and solves the problem, then ask how they might improve their solution based on Time and Space complexity.
    Keep this back and forth of question and answering with hints going until the user has either achieved at least a semi-optimal solution or gives up and asks for the answer.
    """

    summarized_prompt = summarize_conversation([context])
    context_li = [f"Problem: {summarized_prompt}"]


    print(f"\nProblem: {context}")
    initial_input = input(f"\nHow would you go about solving this?\n>>> ")
    while initial_input.strip() == "":
        initial_input = input("\nI'm sorry I didn't get that, could you please try again?\n>>> ")
    
    if "stop" in initial_input.lower() or "exit()" in initial_input.lower(): return

    dialog = [UserMessage(content=initial_input, context=summarize_conversation(context_li)), SystemMessage(content=system_guidance)]
    out_msg = simple_question_answer(generator, dialog)

    context_li.append(f"User: {initial_input}")
    context_li.append(f"Agent: {out_msg.content}")

    user_query = ""

    while True:
        user_query = input("\n>>> ")
        if user_query.strip() == "":
            print(f"\nI'm sorry I didn't get that, could you please try again?")
            continue
        
        if "stop" in user_query.lower() or "exit()" in user_query.lower(): break
        dialog = [UserMessage(content=user_query, context=summarize_conversation(context_li)), SystemMessage(content=system_guidance)]
        out_msg = simple_question_answer(generator, dialog)
        
        context_li.append(f"User: {user_query}")
        context_li.append(f"Agent: {out_msg.content}")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Carter AI",
        description="Example Pair Programming with LLama LLM",
        epilog="Try your best to solve the program with our AI assistant!" 
    )
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help="path to LLamam Model Checkpoint Directory (e.g. ~/.llama/checkpoints/Llama3.2-3B-Instruct)")
    args = parser.parse_args()
    args = vars(args)
    
    if is_valid_args(args):
        checkpoint_path = args['checkpoint']
        run(checkpoint_path)
    else:
        print(f"Invalid Arguments! Please try again.")