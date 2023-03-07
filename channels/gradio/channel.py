# encoding:utf-8

"""
gradio channel
"""
import json
import arrow
import pandas as pd
from config import Config
from common.utils import IntEncoder, MarkdownUtils, Fetcher
from common.data import Reply, Context, Query
from common.log import logger
from channels.channel import Channel
from typing import List, Tuple, Dict, Generator
import openai
import gradio as gr

openai.api_key = Config.openai('api_key')
if Config.openai('use_proxy') and Config.proxy():
    openai.proxy = Config.proxy()

if Config.openai('api_base'):
    openai.api_base = Config.openai('api_base')


class GradioChannel(Channel):
    prompt_templates = {"Default Prompt": ""}

    css = """
          #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
          #chatbox {min-height: 400px;}
          #header {text-align: center;}
          #prompt_template_preview {padding: 1em; border-width: 1px; border-style: solid; border-color: #e0e0e0; border-radius: 4px;}
          #total_tokens_str {text-align: right; font-size: 0.8em; color: #666; height: 1em;}
          #label {font-size: 0.8em; padding: 0.5em; margin: 0;}
          .message { font-size: 1.2em; }
          """

    def __init__(self):
        self.demo = self.__build_blocks()
        # self.demo.queue()

    @property
    def server(self):
        return self.demo

    def __auth(self, username, password):
        users = {
            'tutu': 'maomao',
            'maomao': 'tutu',
        }
        is_login = username in users.keys() and users[username] == password
        return is_login

    @classmethod
    def __empty_state(cls):
        return {
            'total_tokens': 0,
            'total_images': 0,
            'chat_records': [],
            'image_records': [],
        }

    def __download_prompt_templates(self):
        data, _, _ = Fetcher.fetch_file_data(Config.openai('prompt_url'))
        df = pd.read_csv('prompts.csv', sep=',', error_bad_lines=False)
        for i in df.index:
            # update global state prompt_templates
            self.prompt_templates[df.at[i, 'act']] = df.at[i, 'prompt']
        choices = list(self.prompt_templates.keys())
        return gr.update(value=choices[0], choices=choices)

    def __clear_conversation(self):
        return gr.update(value=None, visible=True), None, '', self.__empty_state()

    def __on_prompt_template_change(self, prompt_choice):
        if not isinstance(prompt_choice, str):
            return
        return self.prompt_templates.get(prompt_choice, '')

    def __submit_message(self, prompt, prompt_choice, temperature, max_tokens, state):
        history = state['chat_records']
        if not prompt:
            return gr.update(value='', visible=state['total_tokens'] < 1536), [
                (history[i]['content'], history[i + 1]['content']) for i in
                range(0, len(history) - 1, 2)], f"Total tokens used: {state['total_tokens']} / 4096", state

        prompt_template = self.prompt_templates.get(prompt_choice, '')

        system_prompt = []
        if prompt_template:
            system_prompt = [{"role": "system", "content": prompt_template}]

        prompt_msg = {"role": "user", "content": prompt}

        try:
            _before = arrow.now().float_timestamp
            logger.debug(f'start of ChatCompletion, time={_before}')
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=system_prompt + history + [prompt_msg],
                temperature=temperature, max_tokens=max_tokens
            )
            _after = arrow.now().float_timestamp
            logger.debug(f'end of ChatCompletion, time={_after}, used={_after - _before}')
            history.append(prompt_msg)
            history.append(completion.choices[0].message.to_dict())
            state['total_tokens'] += completion['usage']['total_tokens']
        except Exception as e:
            history.append(prompt_msg)
            history.append({
                "role": "system",
                "content": f"Error: {e}"
            })

        total_tokens_used_msg = f"Total tokens used: {state['total_tokens']} / 4096"
        chat_messages = [(history[i]['content'], history[i + 1]['content']) for i in range(0, len(history) - 1, 2)]
        input_visibility = state['total_tokens'] < 4096

        return gr.update(value='', visible=input_visibility), chat_messages, total_tokens_used_msg, state

    def __build_blocks(self):
        with gr.Blocks(self.css) as demo:
            state = gr.State(self.__empty_state())

            with gr.Column(elem_id="col-container"):
                gr.Markdown("""## 兔兔 ChatGPT""", elem_id="header")
                with gr.Row():
                    with gr.Column():
                        chat_bot = gr.Chatbot(elem_id="chatbox")
                        input_message = gr.Textbox(
                            show_label=False, placeholder="Enter text and press enter", visible=True).style(container=False)
                        btn_submit = gr.Button("Submit")
                        total_tokens_str = gr.Markdown(elem_id="total_tokens_str")
                        btn_clear_conversation = gr.Button("🔃 New Conversation")
                    with gr.Column():
                        prompt_choice = gr.Dropdown(label="Set a custom insruction for the chatbot:",
                                                    choices=list(self.prompt_templates.keys()))
                        prompt_template_preview = gr.Markdown(elem_id="prompt_template_preview")
                        with gr.Accordion("Advanced parameters", open=False):
                            temperature = gr.Slider(minimum=0, maximum=2.0, value=0.7, step=0.1, interactive=True,
                                                    label="Temperature (higher = more creative/chaotic)")
                            max_tokens = gr.Slider(minimum=100, maximum=4096, value=1536, step=1, interactive=True,
                                                   label="Max tokens per response")

            btn_submit.click(self.__submit_message,
                             [input_message, prompt_choice, temperature, max_tokens, state],
                             [input_message, chat_bot, total_tokens_str, state], queue=False)
            input_message.submit(self.__submit_message,
                                 [input_message, prompt_choice, temperature, max_tokens, state],
                                 [input_message, chat_bot, total_tokens_str, state], queue=False)
            btn_clear_conversation.click(self.__clear_conversation, [], [input_message, chat_bot, total_tokens_str, state])
            prompt_choice.change(self.__on_prompt_template_change, inputs=[prompt_choice], outputs=[prompt_template_preview])
            demo.load(self.__download_prompt_templates, inputs=None, outputs=[prompt_choice])
        return demo

    def startup(self):
        self.demo.launch(
            debug=True, share=False, auth=self.__auth, auth_message='请输入用户名和密码: ',
            server_name='0.0.0.0', server_port=Config.gradio('server_port'))
