import sys, logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')
sys.path.insert(0, '.')
from core.llm.llm_interface import LLMInterface

llm = LLMInterface(streaming=False, model='kitsu:character')
print('is_character_model:', llm.is_character_model)
print('prompt_builder type:', type(llm.prompt_builder))

# direct build
try:
    pb_res = None
    if llm.prompt_builder:
        pb_res = llm.prompt_builder.build(user_input='Hi!', emotion='neutral', mood='behave', style='sweet')
        print('PromptBuilder.build result starts:\n', pb_res[:200])
    else:
        print('No prompt_builder available')
except Exception as e:
    print('PromptBuilder.build raised:', e)

# Emulate _build_prompt internals with explicit try/except to capture errors
import traceback
try:
    print('\n--- Emulating character prompt build ---')
    if llm.is_character_model:
        print('is_character_model is True')
        try:
            print('prompt_builder exists?', bool(getattr(llm, 'prompt_builder', None)))
            if getattr(llm, 'prompt_builder', None):
                # Non-greeting path
                user_info = None
                memory_context = None
                if llm.memory:
                    try:
                        user_info = llm.memory.get_user_info()
                    except Exception:
                        user_info = {}
                    try:
                        memory_context = getattr(llm.memory, 'recall', None) and llm.memory.recall(context_length=3)
                    except Exception:
                        memory_context = []

                print('Calling prompt_builder.build...')
                res = llm.prompt_builder.build(
                    user_input=("Hi!"),
                    emotion='neutral',
                    mood='behave',
                    style='sweet',
                    user_info=user_info,
                    memory_context=memory_context
                )
                print('EMULATED BUILD RESULT STARTS:\n', res[:400])
        except Exception as e:
            print('EMULATED BUILD ERROR:', e)
            traceback.print_exc()
    else:
        print('is_character_model is False')
except Exception as e:
    print('Top-level error:', e)
    traceback.print_exc()

