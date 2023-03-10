# encoding:utf-8
import re
import openai
import time
import json
from bot.bot import Bot
from config import Config
from common.utils import MarkdownUtils
from common.log import logger
from common.data import Reply
from .session import SessionManager
from .token import Token
from oss.aliyun_helper import Helper as Oss


class OpenAIBot(Bot):
    config = Config.openai()

    def chat(self, query):
        model = self.config['chat_model']
        messages = [{'role': 'user', 'content': query}]
        tokens = Token.length_messages(messages, model=model)
        max_tokens = Token.max_tokens(model) - Token.length_messages(messages, model=model) - 128
        logger.debug(f'[OPENAI] chat model={model}, tokens={tokens}, message={json.dumps(messages, ensure_ascii=False)}')
        response = openai.ChatCompletion.create(
            model=model,  # 对话模型的名称
            messages=messages,
            temperature=0.9,  # 值在[0,1]之间，越大表示回复越具有不确定性
            max_tokens=max_tokens,  # 回复最大的字符数
            top_p=1,
            frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
        )
        usage = response.usage
        role = response.choices[0].message.role.strip()
        answer = response.choices[0].message.content.strip()
        logger.debug(f'[OPENAI] reply_chat role={role}, answer={answer}, usage={json.dumps(usage, ensure_ascii=False)}')
        return answer

    @classmethod
    def prefix_parser(cls, query, prefix_list) -> (str, str):
        query = query.strip()
        for prefix in prefix_list:
            if query.lower().startswith(prefix.lower()):
                _tmp = re.split(prefix, query, maxsplit=1, flags=re.IGNORECASE)
                if len(_tmp) > 1:
                    return prefix, _tmp[1].strip()
        return None, query

    def reply(self, query, context):
        query = query.msg
        logger.debug(f'[OPENAI] reply query={query}')
        _cmd_prefix, query = self.prefix_parser(query, self.config['cmd_prefix'])
        if _cmd_prefix or query == '':
            return self.reply_cmd(query, context)

        _image_prefix, query = self.prefix_parser(query, self.config['image_prefix'])
        if _image_prefix:
            return self.reply_img(query, context)

        query = query.strip()
        return self.reply_chat(query, context)

    def reply_cmd(self, query, context):
        logger.debug(f'[OPENAI] reply_cmd query={query}')

        sm = SessionManager(context)
        joined_session = sm.joined_session

        if query.startswith('help') or query == '':
            msg = \
                f'命令格式如下: \n' \
                f'/help\n' \
                f'/新建对话%[标题]%[性格]\n' \
                f'/删除对话%<对话ID>\n' \
                f'/对话列表\n' \
                f'/切换对话%<对话ID>\n' \
                f'/重启对话\n' \
                f'/最近对话%[N=0, 限制Tokens={self.config["max_query_tokens"]}的对话; N>0, 最近N条对话; N<0, 全部对话]\n' \
                f'/标题%[新标题]\n' \
                f'/性格%[{self.config["character"]}]\n'
            return Reply(by='openai_cmd', type='TEXT', result='done', msg=msg)

        elif query.startswith('新建对话'):
            _tmp = query.split('%', 2)
            if len(_tmp) == 2:
                new_session_id = sm.new_session(title=_tmp[1])
            elif len(_tmp) == 3:
                new_session_id = sm.new_session(title=_tmp[1], character=_tmp[2])
            else:
                new_session_id = sm.new_session()
            new_session = sm.join_session(new_session_id)
            return Reply(
                by='openai_cmd', type='TEXT', result='done',
                msg=f'新建对话ID: {new_session["session_id"]}, 标题: {new_session["title"]}, 性格: {new_session["character"]}')

        elif query.startswith('删除对话%'):
            _tmp = query.split('%', 1)
            if len(_tmp) == 2:
                session_id = _tmp[1].strip()
                sessions = sm.sessions
                for s in sessions:
                    if s['session_id'] == session_id:
                        if joined_session['session_id'] != session_id:
                            sm.remove_session(session_id)
                            return Reply(by='openai_cmd', type='TEXT', result='done', msg=f'已删除对话: {session_id}')
                        else:
                            return Reply(by='openai_cmd', type='TEXT', result='error', msg=f'无法删除当前对话: {session_id}')
            return Reply(by='openai_cmd', type='TEXT', result='error', msg=f'对话不存在!')

        elif query.startswith('对话列表'):
            sessions = sm.sessions
            msg = ''
            for s in sessions:
                if s['session_id'] == joined_session['session_id']:
                    msg += f'\n>>>>'
                else:
                    msg += f'\n----'
                msg += f'对话ID: {s["session_id"]}, 标题: {s["title"]}, 性格: {s["character"]}'
            return Reply(by='openai_cmd', type='TEXT', result='done', msg=msg)

        elif query.startswith('切换对话%'):
            _tmp = query.split('%', 1)
            if len(_tmp) == 2:
                session_id = _tmp[1].strip()
                sessions = sm.sessions
                for s in sessions:
                    if s['session_id'] == session_id:
                        session = sm.join_session(session_id)
                        return Reply(
                            by='openai_cmd', type='TEXT', result='done',
                            msg=f'切换对话 ID: {session["session_id"]}, 标题: {session["title"]}, 性格: {session["character"]}')
            return Reply(by='openai_cmd', type='TEXT', result='error', msg=f'对话不存在!')

        elif query.startswith('重启对话'):
            sm.reset_records(joined_session['session_id'])
            return Reply(by='openai_cmd', type='TEXT', result='done', msg='对话已重启!')

        elif query.startswith('最近对话'):
            _tmp = query.split('%', 1)
            if len(_tmp) == 2:
                num = int(_tmp[1]) if _tmp[1].isdigit() else -1
                msg = sm.recent_chat_content(joined_session['session_id'], self.config['chat_model'], num=num)
            else:
                msg = sm.recent_chat_content(joined_session['session_id'], self.config['chat_model'], num=0)

            return Reply(by='openai_cmd', type='TEXT', result='done', msg=f'{msg}')

        elif query.startswith('标题'):
            _tmp = query.split('%', 1)
            if len(_tmp) == 2:
                msg = f'对话ID: {joined_session["session_id"]}, 标题: {joined_session["title"]}, 性格: {joined_session["character"]}, 标题修改为: {_tmp[1]}'
                sm.set_session(joined_session['session_id'], title=_tmp[1])
                return Reply(by='openai_cmd', type='TEXT', result='done', msg=msg)
            else:
                msg = joined_session.get('title')
                return Reply(by='openai_cmd', type='TEXT', result='done', msg=f'对话{joined_session["session_id"]}的标题是: {msg}')

        elif query.startswith('性格'):
            _tmp = query.split('%', 1)
            if len(_tmp) == 2:
                msg = f'对话ID: {joined_session["session_id"]}, 标题: {joined_session["title"]}, 性格: {joined_session["character"]}, 性格修改为: {_tmp[1]}'
                sm.set_session(joined_session['session_id'], character=_tmp[1])
                return Reply(by='openai_cmd', type='TEXT', result='done', msg=msg)
            else:
                msg = joined_session.get('character')
                return Reply(by='openai_cmd', type='TEXT', result='done', msg=f'对话{joined_session["session_id"]}的性格是: {msg}')

        else:
            return Reply(by='openai_cmd', type='TEXT', result='error', msg='不支持该命令!')

    def reply_chat(self, query, context, retry_count=0):
        logger.debug(f'[OPENAI] reply_chat query={query}')

        sm = SessionManager(context)
        joined_session = sm.joined_session
        session_id = joined_session['session_id']

        try:
            model = self.config['chat_model']
            messages = sm.build_chat_messages(session_id, query, model)
            tokens = Token.length_messages(messages, model=model)
            max_tokens = Token.max_tokens(model) - tokens - 128
            logger.debug(f'[OPENAI] create chat completion model={model}, tokens={tokens}, message={json.dumps(messages, ensure_ascii=False)}')
            response = openai.ChatCompletion.create(
                model=model,  # 对话模型的名称
                messages=messages,
                temperature=0.9,  # 值在[0,1]之间，越大表示回复越具有不确定性
                max_tokens=max_tokens,  # 回复最大的字符数
                top_p=1,
                frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
                presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            )
            usage = response.usage
            role = response.choices[0].message.role.strip()
            answer = response.choices[0].message.content.strip()
            logger.debug(f'[OPENAI] reply_chat role={role}, answer={answer}, usage={json.dumps(usage, ensure_ascii=False)}')
            sm.add_record(session_id, query, answer)
            image_urls = MarkdownUtils.extract_images(answer)
            if image_urls:
                oss_urls = [Oss.upload_url(i) for i in image_urls]
                return Reply(by=f'reply_chat', type='IMAGES', result='done', msg=oss_urls)
            return Reply(by=f'reply_chat', type='TEXT', result='done', msg=answer)

        except openai.error.InvalidRequestError as e:
            logger.warn(e)
            return Reply(by=f'reply_chat', type='TEXT', result='error', msg=f'对话上下文超过最大Token限制，建议重启对话')

        except openai.error.RateLimitError as e:
            # rate limit exception
            logger.warn(e)
            if retry_count < self.config['retry_times']:
                time.sleep(self.config['retry_interval'])
                logger.warn(f'[OPENAI] completion rate limit exceed, 第{retry_count+1}次重试')
                return self.reply_chat(query, context, retry_count+1)
            else:
                return Reply(by=f'reply_chat', type='TEXT', result='error', msg='提问太快啦，请休息一下再问我吧!')

        except Exception as e:
            # unknown exception
            logger.exception(e)
            return Reply(by=f'reply_chat', type='TEXT', result='error', msg='OpenAI出小差了, 请再问我一次吧!')

    def reply_img(self, query, context, retry_count=0):
        logger.debug(f'[OPENAI] reply_img query={query}')

        try:
            prompt = query
            response = openai.Image.create(
                prompt=prompt,  # 图片描述
                n=1,             # 每次生成图片的数量
                size="512x512"   # 图片大小,可选有 256x256, 512x512, 1024x1024
            )
            image_url = response['data'][0]['url']
            image_url = Oss.upload_url(image_url)
            logger.debug(f'[OPENAI] reply_img answer={image_url}')
            return Reply(by='openai_img', type='IMAGE', result='done', msg=image_url)
        except openai.error.RateLimitError as e:
            logger.warn(e)
            if retry_count < self.config['retry_times']:
                time.sleep(self.config['retry_interval'])
                logger.warn(f'[OPENAI] rate limit exceed, 第{retry_count+1}次重试')
                return self.reply_img(query, context, retry_count+1)
            else:
                return Reply(by='openai_img', type='TEXT', result='error', msg='提问太快啦，请休息一下再问我吧!')
        except Exception as e:
            logger.exception(e)
            return Reply(by='openai_img', type='TEXT', result='error', msg='图片生成失败, 请再问我一次吧!')


openai.api_key = Config.openai('api_key')

if Config.openai('use_proxy') and Config.proxy():
    openai.proxy = Config.proxy()

if Config.openai('api_base'):
    openai.api_base = Config.openai('api_base')
